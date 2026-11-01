"""CLI script to run single-video anomaly inference, export timestamped CSVs, timeline plots, and overlays."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vad.inference.predictor import VideoAnomalyPredictor
from vad.inference.visualization import export_annotated_video, plot_anomaly_timeline
from vad.utils.logging import setup_logger

logger = setup_logger("vad.predict_video")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sliding-window anomaly inference on a local video file."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file or frame directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained AnomalyDetector checkpoint (.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Directory to save CSV scores, timeline PNG, and optional annotated MP4.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="Sliding window length in frames (default from checkpoint config).",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Sliding window stride in frames (default from checkpoint config).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Anomaly visualization threshold in (0, 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device override (auto, cpu, cuda, directml, mps).",
    )
    parser.add_argument(
        "--export-video",
        action="store_true",
        help="Export an annotated MP4 video with live anomaly score overlays.",
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

    predictor = VideoAnomalyPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device_override=args.device,
        use_synthetic_encoders=args.synthetic_encoders,
    )

    logger.info(
        "Running video inference on %s (device=%s, synthetic_mode=%s)...",
        args.video,
        predictor.device,
        predictor.use_synthetic_encoders or predictor.checkpoint_is_synthetic,
    )

    result, decoded = predictor.predict_video_file(
        video_path=args.video,
        window_length=args.window_length,
        window_stride=args.window_stride,
        threshold=args.threshold,
    )

    out_dir = Path(args.output_dir)
    clip_csv, frame_csv = result.save_csv(out_dir)

    plot_path = out_dir / f"{result.video_id}_timeline.png"
    fig = plot_anomaly_timeline(result, output_path=plot_path)
    plt.close(fig)

    logger.info("Saved clip-level scores CSV  : %s", clip_csv)
    logger.info("Saved frame-level scores CSV : %s", frame_csv)
    logger.info("Saved timeline figure        : %s", plot_path)

    if args.export_video:
        annotated_path = out_dir / f"{result.video_id}_annotated.mp4"
        export_annotated_video(decoded.frames_rgb, result, annotated_path)
        logger.info("Saved annotated overlay video: %s", annotated_path)

    print("=" * 72)
    print(f"VIDEO PREDICTION SUMMARY — {result.video_id}")
    print("=" * 72)
    if result.is_synthetic_mode:
        print("[NOTICE] Executed in Synthetic Smoke-Test Mode (no pretrained HF weights used).")
    print(f"Frames / Duration : {result.total_frames} frames ({result.duration_sec:.2f}s @ {result.fps:.1f} FPS)")
    print(f"Sliding Windows   : {len(result.windows)} clips")
    print(f"Peak Anomaly Score: {float(result.smoothed_clip_scores.max()):.4f}")
    print(f"Mean Anomaly Score: {float(result.smoothed_clip_scores.mean()):.4f}")
    print(f"Inference Time    : {result.inference_time_sec:.3f}s ({result.effective_fps:.1f} FPS)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
