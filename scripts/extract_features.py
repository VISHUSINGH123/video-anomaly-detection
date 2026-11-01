"""CLI script to extract and cache aligned VideoMAE and DINOv2 clip features from video files."""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vad.config import load_config
from vad.data.annotations import VideoAnnotation, align_annotations_to_windows, load_annotations_directory
from vad.data.dataset import CachedVideoFeatures, save_cached_features
from vad.data.video_loader import build_temporal_windows, decode_video, discover_videos
from vad.device import resolve_device
from vad.inference.predictor import _extract_deterministic_synthetic_features
from vad.models.dinov2_encoder import DINOv2Encoder
from vad.models.videomae_encoder import VideoMAEEncoder
from vad.utils.logging import setup_logger
from vad.utils.reproducibility import set_seed

logger = setup_logger("vad.extract_features")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract VideoMAE and DINOv2 features from videos and cache aligned .npz bundles."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory of input videos (overrides config.data.raw_video_dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write cached .npz feature files (overrides config.data.processed_feature_dir).",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default=None,
        help="Directory or file containing ground-truth annotations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (auto, cpu, cuda, directml, mps).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract and overwrite existing cached .npz files.",
    )
    parser.add_argument(
        "--synthetic-encoders",
        action="store_true",
        help="Use deterministic synthetic frame statistics instead of downloading Hugging Face checkpoints (for smoke testing only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    video_dir = Path(args.video_dir or cfg.data.raw_video_dir)
    output_dir = Path(args.output_dir or cfg.data.processed_feature_dir)
    ann_dir = Path(args.annotations_dir or cfg.data.annotations_dir)
    dev_info = resolve_device(args.device or cfg.device)

    videos = discover_videos(video_dir)
    if not videos:
        logger.error("No supported video files or frame directories found in %s", video_dir)
        return 1

    annotations = load_annotations_directory(ann_dir) if ann_dir.exists() else {}
    logger.info(
        "Discovered %d videos in %s | Loaded %d annotations | Device: %s",
        len(videos),
        video_dir,
        len(annotations),
        dev_info.device_name,
    )

    videomae_enc: VideoMAEEncoder | None = None
    dinov2_enc: DINOv2Encoder | None = None

    if not args.synthetic_encoders:
        if cfg.encoders.videomae.enabled:
            videomae_enc = VideoMAEEncoder(cfg.encoders.videomae, device=dev_info.device)
            videomae_enc.load_pretrained()
        if cfg.encoders.dinov2.enabled:
            dinov2_enc = DINOv2Encoder(cfg.encoders.dinov2, device=dev_info.device)
            dinov2_enc.load_pretrained()
    else:
        logger.warning(
            "Running in --synthetic-encoders smoke-test mode. Output bundles will be marked is_synthetic=True."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    for vid_path in tqdm(videos, desc="Extracting features"):
        vid_id = vid_path.name if vid_path.is_dir() else vid_path.stem
        out_file = output_dir / f"{vid_id}.npz"
        if out_file.exists() and not args.overwrite:
            logger.info("Skipping %s (already cached at %s)", vid_id, out_file)
            processed_count += 1
            continue

        decoded = decode_video(vid_path, target_fps=cfg.data.target_fps)
        clips_rgb, windows = build_temporal_windows(
            decoded,
            clip_length=cfg.data.clip_length,
            clip_stride=cfg.data.clip_stride,
            short_video_policy=cfg.data.short_video_policy,
        )
        if len(windows) == 0:
            logger.warning("Skipping %s: insufficient frames for clip_length=%d", vid_id, cfg.data.clip_length)
            continue

        if args.synthetic_encoders:
            v_feats, d_feats = _extract_deterministic_synthetic_features(
                clips_rgb, feature_dim=cfg.encoders.videomae.feature_dim
            )
        else:
            v_feats = (
                videomae_enc.extract_clips(clips_rgb)
                if videomae_enc is not None
                else None
            )
            d_feats = (
                dinov2_enc.extract_clips(clips_rgb)[0]
                if dinov2_enc is not None
                else None
            )

        if vid_id in annotations:
            ann = annotations[vid_id]
        else:
            # Infer split from parent folder name if present (e.g., Train001 vs Test001)
            inferred_split = "train" if "train" in vid_id.lower() or "train" in str(vid_path).lower() else "test"
            ann = VideoAnnotation(
                video_id=vid_id,
                split=inferred_split,
                video_label=0,
                num_frames=decoded.original_num_frames,
            )

        bin_labels, ratios = align_annotations_to_windows(
            ann, windows, total_frames=decoded.original_num_frames
        )

        bundle = CachedVideoFeatures(
            video_id=vid_id,
            split=ann.split,
            fps=decoded.fps,
            original_num_frames=decoded.original_num_frames,
            windows=windows,
            videomae_features=v_feats,
            dinov2_features=d_feats,
            clip_binary_labels=bin_labels,
            clip_anomaly_ratios=ratios,
            video_label=ann.video_label,
            is_synthetic=bool(args.synthetic_encoders or "synthetic" in vid_id.lower()),
        )
        save_cached_features(out_file, bundle)
        processed_count += 1

    logger.info("Successfully extracted and cached %d video feature bundles in %s", processed_count, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
