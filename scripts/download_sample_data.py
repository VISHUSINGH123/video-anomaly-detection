"""Dataset setup verification and explicit synthetic sample video/feature generator for smoke testing.

IMPORTANT:
- This script NEVER fabricates real surveillance benchmark data.
- When `--mode check` is used, it inspects `data/raw` and `data/annotations` for genuine datasets
  (UCSD Ped2, ShanghaiTech, UCF-Crime) and prints verification statistics or download instructions.
- When `--mode synthetic` is used, it generates explicitly labeled synthetic surveillance-style
  videos (normal steady pedestrian motion vs. anomalous high-speed intrusion/flash objects) and
  corresponding `.npz` synthetic features for end-to-end pipeline verification.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vad.data.annotations import VideoAnnotation, align_annotations_to_windows, load_annotations_directory
from vad.data.dataset import CachedVideoFeatures, save_cached_features
from vad.data.video_loader import build_temporal_windows, decode_video, discover_videos
from vad.inference.predictor import _extract_deterministic_synthetic_features
from vad.utils.logging import setup_logger
from vad.utils.reproducibility import set_seed

logger = setup_logger("vad.download_sample_data")


def _render_synthetic_surveillance_video(
    output_path: Path,
    num_frames: int = 64,
    fps: float = 16.0,
    width: int = 224,
    height: int = 224,
    anomaly_interval: Tuple[int, int] | None = None,
    seed: int = 42,
) -> None:
    """Render a synthetic top-down walkway video.

    Normal frames contain slow, steady circular pedestrian blobs walking left-to-right.
    Anomalous frames (inside `anomaly_interval`) contain a high-speed bright red/yellow vehicle-like
    rectangle racing diagonally across the scene with high motion velocity.
    """
    rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    ped1_x = int(rng.integers(20, 60))
    ped2_x = int(rng.integers(100, 150))

    try:
        for f_idx in range(num_frames):
            # Textured gray pavement background
            frame = np.full((height, width, 3), 55, dtype=np.uint8)
            noise = rng.integers(-4, 5, size=(height, width, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Sidewalk lane lines
            cv2.line(frame, (0, 60), (width, 60), (95, 95, 95), 2)
            cv2.line(frame, (0, 165), (width, 165), (95, 95, 95), 2)

            # Normal slow-moving pedestrians (speed = 1 px/frame)
            p1_x = (ped1_x + f_idx) % width
            p2_x = (ped2_x + f_idx) % width
            cv2.circle(frame, (p1_x, 95), 9, (180, 190, 180), -1)
            cv2.circle(frame, (p2_x, 130), 9, (170, 180, 195), -1)

            # Inject synthetic anomaly if within interval
            if anomaly_interval is not None:
                s_f, e_f = anomaly_interval
                if s_f <= f_idx <= e_f:
                    rel = f_idx - s_f
                    fast_x = (20 + rel * 14) % (width - 30)
                    fast_y = (35 + rel * 9) % (height - 30)
                    # High-contrast fast anomalous object + localized intensity flash
                    cv2.rectangle(
                        frame,
                        (fast_x, fast_y),
                        (fast_x + 34, fast_y + 24),
                        (20, 40, 250),
                        -1,
                    )
                    cv2.rectangle(
                        frame,
                        (max(0, fast_x - 10), max(0, fast_y - 10)),
                        (min(width - 1, fast_x + 44), min(height - 1, fast_y + 34)),
                        (40, 230, 250),
                        2,
                    )

            writer.write(frame)
    finally:
        writer.release()


def generate_synthetic_dataset(
    raw_dir: Path,
    processed_dir: Path,
    annotations_dir: Path,
    clip_length: int = 16,
    clip_stride: int = 8,
    feature_dim: int = 768,
    seed: int = 42,
) -> Dict[str, Path]:
    """Generate explicitly labeled synthetic videos, annotations, and cached `.npz` feature bundles."""
    set_seed(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        # (video_id, split, num_frames, anomaly_interval)
        ("synthetic_train_normal_01", "train", 64, None),
        ("synthetic_train_normal_02", "train", 64, None),
        ("synthetic_train_anom_01", "train", 64, (16, 47)),
        ("synthetic_train_anom_02", "train", 64, (24, 55)),
        ("synthetic_val_normal_01", "val", 64, None),
        ("synthetic_val_anom_01", "val", 64, (16, 47)),
        ("synthetic_test_normal_01", "test", 64, None),
        ("synthetic_test_anom_01", "test", 64, (16, 47)),
    ]

    annotations_payload: Dict[str, dict] = {"_WARNING": "SYNTHETIC SMOKE-TEST DATA ONLY", "videos": {}}
    rng = np.random.default_rng(seed)

    for idx, (vid_id, split, n_frames, interval) in enumerate(specs):
        vid_path = raw_dir / f"{vid_id}.mp4"
        _render_synthetic_surveillance_video(
            output_path=vid_path,
            num_frames=n_frames,
            fps=16.0,
            anomaly_interval=interval,
            seed=seed + idx * 17,
        )

        intervals_list = [list(interval)] if interval is not None else []
        vid_label = 1 if interval is not None else 0
        annotations_payload["videos"][vid_id] = {
            "split": split,
            "video_label": vid_label,
            "num_frames": n_frames,
            "anomaly_intervals": intervals_list,
            "category": "synthetic_anomaly" if vid_label == 1 else "synthetic_normal",
        }

        ann = VideoAnnotation(
            video_id=vid_id,
            split=split,
            video_label=vid_label,
            num_frames=n_frames,
            anomaly_intervals=[interval] if interval is not None else [],
            category="synthetic_anomaly" if vid_label == 1 else "synthetic_normal",
        )

        decoded = decode_video(vid_path)
        clips_rgb, windows = build_temporal_windows(
            decoded, clip_length=clip_length, clip_stride=clip_stride
        )
        bin_labels, ratios = align_annotations_to_windows(
            ann, windows, total_frames=n_frames
        )

        # Extract deterministic visual statistics from the rendered frames + small noise
        v_feats, d_feats = _extract_deterministic_synthetic_features(
            clips_rgb, feature_dim=feature_dim
        )
        v_feats = v_feats + rng.normal(0.0, 0.05, size=v_feats.shape).astype(np.float32)
        d_feats = d_feats + rng.normal(0.0, 0.05, size=d_feats.shape).astype(np.float32)

        bundle = CachedVideoFeatures(
            video_id=vid_id,
            split=split,
            fps=decoded.fps,
            original_num_frames=decoded.original_num_frames,
            windows=windows,
            videomae_features=v_feats,
            dinov2_features=d_feats,
            clip_binary_labels=bin_labels,
            clip_anomaly_ratios=ratios,
            video_label=vid_label,
            is_synthetic=True,
        )
        save_cached_features(processed_dir / f"{vid_id}.npz", bundle)

    ann_file = annotations_dir / "synthetic_annotations.json"
    with open(ann_file, "w", encoding="utf-8") as f:
        json.dump(annotations_payload, f, indent=2)

    logger.info(
        "Generated %d synthetic smoke-test videos in %s and cached features in %s",
        len(specs),
        raw_dir,
        processed_dir,
    )
    return {
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "annotations_file": ann_file,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify real dataset directories or generate explicitly labeled synthetic smoke-test data."
    )
    parser.add_argument(
        "--mode",
        choices=["check", "synthetic"],
        default="check",
        help="'check' verifies existing datasets; 'synthetic' creates labeled synthetic smoke-test data.",
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw/synthetic")
    parser.add_argument("--processed-dir", type=str, default="data/processed/synthetic")
    parser.add_argument("--annotations-dir", type=str, default="data/annotations/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "check":
        raw_p = Path(args.raw_dir)
        ann_p = Path(args.annotations_dir)
        videos = discover_videos(raw_p) if raw_p.exists() else []
        anns = load_annotations_directory(ann_p) if ann_p.exists() else {}
        print(f"Discovered {len(videos)} videos in {raw_p}")
        print(f"Discovered {len(anns)} annotations in {ann_p}")
        if not videos:
            print(
                "No real surveillance videos found yet. See docs/dataset_setup.md for official "
                "UCSD Pedestrian, ShanghaiTech, and UCF-Crime setup instructions, or run with "
                "`--mode synthetic` to create synthetic smoke-test data."
            )
        return 0

    generate_synthetic_dataset(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        annotations_dir=Path(args.annotations_dir),
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
