"""Feature caching and PyTorch Dataset for temporally aligned multi-stream video features."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from vad.data.annotations import VideoAnnotation, align_annotations_to_windows
from vad.data.video_loader import TemporalWindow


@dataclass
class CachedVideoFeatures:
    """Disk-cached feature bundle for a single video."""

    video_id: str
    split: str
    fps: float
    original_num_frames: int
    windows: List[TemporalWindow]
    videomae_features: Optional[np.ndarray]  # Shape: (T_clips, D_v)
    dinov2_features: Optional[np.ndarray]  # Shape: (T_clips, D_d)
    clip_binary_labels: np.ndarray  # Shape: (T_clips,)
    clip_anomaly_ratios: np.ndarray  # Shape: (T_clips,)
    video_label: int
    is_synthetic: bool = False


def save_cached_features(cache_path: Path | str, bundle: CachedVideoFeatures) -> Path:
    """Save extracted video features, timestamps, and labels to a compressed `.npz` archive."""
    path = Path(cache_path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "video_id": bundle.video_id,
        "split": bundle.split,
        "fps": float(bundle.fps),
        "original_num_frames": int(bundle.original_num_frames),
        "video_label": int(bundle.video_label),
        "is_synthetic": bool(bundle.is_synthetic),
        "windows": [asdict(w) for w in bundle.windows],
    }

    arrays: Dict[str, Any] = {
        "metadata_json": np.array(json.dumps(metadata)),
        "clip_binary_labels": np.asarray(bundle.clip_binary_labels, dtype=np.float32),
        "clip_anomaly_ratios": np.asarray(bundle.clip_anomaly_ratios, dtype=np.float32),
    }
    if bundle.videomae_features is not None:
        arrays["videomae_features"] = np.asarray(bundle.videomae_features, dtype=np.float32)
    if bundle.dinov2_features is not None:
        arrays["dinov2_features"] = np.asarray(bundle.dinov2_features, dtype=np.float32)

    np.savez_compressed(path, **arrays)
    return path


def load_cached_features(cache_path: Path | str) -> CachedVideoFeatures:
    """Load a CachedVideoFeatures bundle from a `.npz` archive and validate alignment."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cached feature file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"].item()))
        windows = [TemporalWindow(**w) for w in meta["windows"]]
        num_clips = len(windows)

        v_feat = data["videomae_features"].astype(np.float32) if "videomae_features" in data else None
        d_feat = data["dinov2_features"].astype(np.float32) if "dinov2_features" in data else None
        bin_labels = data["clip_binary_labels"].astype(np.float32)
        ratios = data["clip_anomaly_ratios"].astype(np.float32)

        if v_feat is not None and v_feat.shape[0] != num_clips:
            raise ValueError(
                f"Temporal misalignment in {path.name}: videomae_features has {v_feat.shape[0]} "
                f"clips, but metadata has {num_clips} windows."
            )
        if d_feat is not None and d_feat.shape[0] != num_clips:
            raise ValueError(
                f"Temporal misalignment in {path.name}: dinov2_features has {d_feat.shape[0]} "
                f"clips, but metadata has {num_clips} windows."
            )

        return CachedVideoFeatures(
            video_id=meta["video_id"],
            split=meta.get("split", "train"),
            fps=float(meta["fps"]),
            original_num_frames=int(meta["original_num_frames"]),
            windows=windows,
            videomae_features=v_feat,
            dinov2_features=d_feat,
            clip_binary_labels=bin_labels,
            clip_anomaly_ratios=ratios,
            video_label=int(meta["video_label"]),
            is_synthetic=bool(meta.get("is_synthetic", False)),
        )


class CachedFeatureDataset(Dataset):
    """PyTorch Dataset over cached video feature sequences for temporal anomaly learning."""

    def __init__(
        self,
        feature_dir: Path | str,
        split: Optional[str] = None,
        videomae_dim: int = 768,
        dinov2_dim: int = 768,
    ) -> None:
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.videomae_dim = videomae_dim
        self.dinov2_dim = dinov2_dim

        if not self.feature_dir.exists():
            raise FileNotFoundError(f"Processed feature directory not found: {self.feature_dir}")

        all_files = sorted(self.feature_dir.rglob("*.npz"))
        self.bundles: List[CachedVideoFeatures] = []
        for f in all_files:
            bundle = load_cached_features(f)
            if split is None or bundle.split.lower() == split.lower():
                self.bundles.append(bundle)

        if not self.bundles:
            raise ValueError(
                f"No cached `.npz` feature bundles found in {self.feature_dir} "
                f"for split={split!r}."
            )

    def __len__(self) -> int:
        return len(self.bundles)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        b = self.bundles[index]
        num_clips = len(b.windows)

        if b.videomae_features is not None:
            v_tensor = torch.from_numpy(b.videomae_features).float()
        else:
            v_tensor = torch.zeros((num_clips, self.videomae_dim), dtype=torch.float32)

        if b.dinov2_features is not None:
            d_tensor = torch.from_numpy(b.dinov2_features).float()
        else:
            d_tensor = torch.zeros((num_clips, self.dinov2_dim), dtype=torch.float32)

        return {
            "video_id": b.video_id,
            "split": b.split,
            "videomae_features": v_tensor,  # (T, D_v)
            "dinov2_features": d_tensor,  # (T, D_d)
            "clip_labels": torch.from_numpy(b.clip_binary_labels).float(),  # (T,)
            "clip_ratios": torch.from_numpy(b.clip_anomaly_ratios).float(),  # (T,)
            "video_label": torch.tensor(float(b.video_label), dtype=torch.float32),
            "length": num_clips,
            "fps": b.fps,
            "original_num_frames": b.original_num_frames,
            "windows": b.windows,
            "is_synthetic": b.is_synthetic,
        }


def collate_variable_length_features(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length clip sequences within a minibatch and create a boolean validity mask.

    Args:
        batch: List of items from CachedFeatureDataset.

    Returns:
        Dictionary containing padded tensors and a `mask` tensor of shape (B, T_max) where True = valid clip.
    """
    batch_size = len(batch)
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)

    v_dim = batch[0]["videomae_features"].shape[-1]
    d_dim = batch[0]["dinov2_features"].shape[-1]

    v_padded = torch.zeros((batch_size, max_len, v_dim), dtype=torch.float32)
    d_padded = torch.zeros((batch_size, max_len, d_dim), dtype=torch.float32)
    labels_padded = torch.zeros((batch_size, max_len), dtype=torch.float32)
    ratios_padded = torch.zeros((batch_size, max_len), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    video_labels = torch.zeros((batch_size,), dtype=torch.float32)

    video_ids: List[str] = []
    windows_list: List[List[TemporalWindow]] = []
    fps_list: List[float] = []
    orig_frames_list: List[int] = []
    synthetic_flags: List[bool] = []

    for i, item in enumerate(batch):
        length = item["length"]
        v_padded[i, :length] = item["videomae_features"]
        d_padded[i, :length] = item["dinov2_features"]
        labels_padded[i, :length] = item["clip_labels"]
        ratios_padded[i, :length] = item["clip_ratios"]
        mask[i, :length] = True
        video_labels[i] = item["video_label"]
        video_ids.append(item["video_id"])
        windows_list.append(item["windows"])
        fps_list.append(item["fps"])
        orig_frames_list.append(item["original_num_frames"])
        synthetic_flags.append(bool(item["is_synthetic"]))

    return {
        "video_ids": video_ids,
        "videomae_features": v_padded,
        "dinov2_features": d_padded,
        "clip_labels": labels_padded,
        "clip_ratios": ratios_padded,
        "video_labels": video_labels,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "windows": windows_list,
        "fps": fps_list,
        "original_num_frames": orig_frames_list,
        "is_synthetic": any(synthetic_flags),
    }
