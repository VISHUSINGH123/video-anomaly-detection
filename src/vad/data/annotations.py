"""Ground-truth annotation parsing, validation, and temporal clip alignment.

Supports:
1. JSON / CSV manifest annotations (frame intervals, frame masks, or video-level labels).
2. ShanghaiTech frame mask `.npy` files (`test_frame_mask/*.npy`).
3. UCSD Pedestrian MATLAB-style `.m` interval scripts (`Ped2.m` / `TestXXX_gt` folders).
4. UCF-Crime temporal annotation `.txt` files (`Temporal_Anomaly_Annotation_for_Testing_Videos.txt`).
"""

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vad.data.video_loader import TemporalWindow


@dataclass
class VideoAnnotation:
    """Validated ground-truth annotation for a single video."""

    video_id: str
    split: str = "test"  # "train", "val", "test"
    video_label: int = 0  # 0 = normal, 1 = anomalous
    num_frames: Optional[int] = None
    anomaly_intervals: List[Tuple[int, int]] = field(default_factory=list)  # 0-based inclusive (start, end)
    frame_labels: Optional[np.ndarray] = None  # Shape (N,), values in {0, 1}
    category: str = "normal"

    def get_frame_labels(self, total_frames: int) -> np.ndarray:
        """Return a binary array of length `total_frames` with 1 for anomalous frames and 0 for normal."""
        if total_frames < 1:
            raise ValueError(f"total_frames must be >= 1, got {total_frames}")

        if self.frame_labels is not None:
            arr = np.asarray(self.frame_labels, dtype=np.int64).reshape(-1)
            if len(arr) >= total_frames:
                return arr[:total_frames]
            padded = np.zeros(total_frames, dtype=np.int64)
            padded[: len(arr)] = arr
            return padded

        labels = np.zeros(total_frames, dtype=np.int64)
        for start_f, end_f in self.anomaly_intervals:
            s = max(0, int(start_f))
            e = min(total_frames - 1, int(end_f))
            if s <= e:
                labels[s : e + 1] = 1
        return labels


def align_annotations_to_windows(
    annotation: VideoAnnotation,
    windows: Sequence[TemporalWindow],
    total_frames: int,
    overlap_threshold: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align frame-level ground-truth annotations to fixed-length temporal windows.

    Args:
        annotation: VideoAnnotation instance.
        windows: Sequence of TemporalWindow objects for the video.
        total_frames: Total number of frames in the video.
        overlap_threshold: Fraction of anomalous frames within a clip required to mark the clip binary label as 1.

    Returns:
        Tuple of:
          - clip_binary_labels: shape (num_windows,), values in {0.0, 1.0}
          - clip_anomaly_ratios: shape (num_windows,), continuous fraction of anomalous frames in [0.0, 1.0]
    """
    if not windows:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    frame_gt = annotation.get_frame_labels(total_frames)
    binary_labels = np.zeros(len(windows), dtype=np.float32)
    ratios = np.zeros(len(windows), dtype=np.float32)

    for idx, win in enumerate(windows):
        valid_indices = [min(max(0, f), total_frames - 1) for f in win.frame_indices]
        clip_gt = frame_gt[valid_indices]
        ratio = float(np.mean(clip_gt)) if len(clip_gt) > 0 else 0.0
        ratios[idx] = ratio
        binary_labels[idx] = 1.0 if ratio >= overlap_threshold else 0.0

    # If only video-level supervision is available (no frame intervals and no frame_labels array)
    if (
        annotation.video_label == 1
        and not annotation.anomaly_intervals
        and annotation.frame_labels is None
    ):
        # In weakly-supervised MIL mode, individual clip ground-truth ratios are unknown;
        # we mark clip_binary_labels with video_label for bag-level training when needed.
        binary_labels[:] = 1.0

    return binary_labels, ratios


def load_annotations_from_json(json_path: Path | str) -> Dict[str, VideoAnnotation]:
    """Load annotations from a structured JSON file.

    Expected JSON structure:
    {
      "videos": {
        "Test001": {
          "split": "test",
          "video_label": 1,
          "num_frames": 180,
          "anomaly_intervals": [[60, 151]],
          "category": "biker"
        }
      }
    }
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    videos_dict = data.get("videos", data)
    result: Dict[str, VideoAnnotation] = {}
    for vid_id, entry in videos_dict.items():
        intervals = [
            (int(iv[0]), int(iv[1])) for iv in entry.get("anomaly_intervals", [])
        ]
        frame_labels_raw = entry.get("frame_labels", None)
        frame_labels = (
            np.asarray(frame_labels_raw, dtype=np.int64)
            if frame_labels_raw is not None
            else None
        )
        video_label = int(
            entry.get(
                "video_label",
                1 if (intervals or (frame_labels is not None and np.any(frame_labels > 0))) else 0,
            )
        )
        result[vid_id] = VideoAnnotation(
            video_id=vid_id,
            split=str(entry.get("split", "test")),
            video_label=video_label,
            num_frames=entry.get("num_frames", None),
            anomaly_intervals=intervals,
            frame_labels=frame_labels,
            category=str(entry.get("category", "anomalous" if video_label == 1 else "normal")),
        )
    return result


def parse_ucsd_matlab_annotations(m_file_path: Path | str) -> Dict[str, VideoAnnotation]:
    """Parse UCSD Pedestrian `.m` ground-truth script (e.g. `UCSDped2.m`).

    Lines in UCSD `.m` files follow the pattern:
    `TestVideoFile{end+1}.gt_frame = [61:180];` (1-based indexing in MATLAB).
    Converted to 0-based inclusive `(start_frame, end_frame)` intervals.
    """
    path = Path(m_file_path)
    if not path.exists():
        raise FileNotFoundError(f"UCSD MATLAB annotation file not found: {path}")

    content = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"gt_frame\s*=\s*\[([^\]]*)\]")
    matches = pattern.findall(content)

    annotations: Dict[str, VideoAnnotation] = {}
    for idx, match_str in enumerate(matches, start=1):
        vid_id = f"Test{idx:03d}"
        intervals: List[Tuple[int, int]] = []
        # Find ranges like 61:180, 95:180
        range_matches = re.findall(r"(\d+)\s*:\s*(\d+)", match_str)
        for s_str, e_str in range_matches:
            # Convert 1-based MATLAB indices to 0-based Python indices
            s_zero = max(0, int(s_str) - 1)
            e_zero = max(s_zero, int(e_str) - 1)
            intervals.append((s_zero, e_zero))

        annotations[vid_id] = VideoAnnotation(
            video_id=vid_id,
            split="test",
            video_label=1 if intervals else 0,
            anomaly_intervals=intervals,
            category="anomalous" if intervals else "normal",
        )
    return annotations


def load_annotations_directory(annotations_dir: Path | str) -> Dict[str, VideoAnnotation]:
    """Load all available annotations from a directory (JSON, CSV, `.m`, or `.npy` masks)."""
    ann_dir = Path(annotations_dir)
    if not ann_dir.exists():
        return {}

    if ann_dir.is_file():
        if ann_dir.suffix.lower() == ".json":
            return load_annotations_from_json(ann_dir)
        elif ann_dir.suffix.lower() == ".m":
            return parse_ucsd_matlab_annotations(ann_dir)

    combined: Dict[str, VideoAnnotation] = {}

    # 1. Check JSON files
    for json_file in sorted(ann_dir.glob("*.json")):
        combined.update(load_annotations_from_json(json_file))

    # 2. Check UCSD .m files
    for m_file in sorted(ann_dir.glob("*.m")):
        combined.update(parse_ucsd_matlab_annotations(m_file))

    # 3. Check ShanghaiTech-style frame mask .npy files
    for npy_file in sorted(ann_dir.rglob("*.npy")):
        vid_id = npy_file.stem
        arr = np.load(npy_file).astype(np.int64).reshape(-1)
        vid_label = int(np.any(arr > 0))
        combined[vid_id] = VideoAnnotation(
            video_id=vid_id,
            split="test",
            video_label=vid_label,
            num_frames=len(arr),
            frame_labels=arr,
            category="anomalous" if vid_label == 1 else "normal",
        )

    # 4. Check CSV manifest files
    for csv_file in sorted(ann_dir.glob("*.csv")):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid_id = row["video_id"].strip()
                split = row.get("split", "test").strip()
                video_label = int(row.get("video_label", 0))
                intervals: List[Tuple[int, int]] = []
                if row.get("start_frame") and row.get("end_frame"):
                    s_f = int(row["start_frame"])
                    e_f = int(row["end_frame"])
                    if s_f >= 0 and e_f >= s_f:
                        intervals.append((s_f, e_f))
                combined[vid_id] = VideoAnnotation(
                    video_id=vid_id,
                    split=split,
                    video_label=video_label,
                    anomaly_intervals=intervals,
                    category=row.get("category", "anomalous" if video_label == 1 else "normal"),
                )

    return combined
