"""CLI script to evaluate a trained AnomalyDetector checkpoint on cached features and annotations."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vad.config import load_config
from vad.data.annotations import load_annotations_directory
from vad.data.dataset import CachedFeatureDataset
from vad.device import resolve_device
from vad.evaluation.evaluator import Evaluator
from vad.training.checkpoints import load_checkpoint
from vad.utils.logging import setup_logger

logger = setup_logger("vad.evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Video Anomaly Detection checkpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=None,
        help="Directory of cached .npz features (overrides config.data.processed_feature_dir).",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default=None,
        help="Directory or file containing ground-truth annotations.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (test, val, train, or all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write JSON and CSV evaluation summaries.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binary classification decision threshold in (0, 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (auto, cpu, cuda, directml, mps).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    dev_info = resolve_device(args.device or cfg.device)

    model, ckpt_cfg, ckpt_meta = load_checkpoint(args.checkpoint, device=dev_info.device)
    if args.threshold is not None:
        ckpt_cfg.evaluation.threshold = args.threshold

    feature_dir = Path(args.feature_dir or cfg.data.processed_feature_dir)
    ann_dir = Path(args.annotations_dir or cfg.data.annotations_dir)
    output_dir = Path(args.output_dir or cfg.evaluation.output_dir)
    split_filter = None if args.split.lower() == "all" else args.split

    dataset = CachedFeatureDataset(
        feature_dir=feature_dir,
        split=split_filter,
        videomae_dim=ckpt_cfg.encoders.videomae.feature_dim,
        dinov2_dim=ckpt_cfg.encoders.dinov2.feature_dim,
    )
    annotations = load_annotations_directory(ann_dir) if ann_dir.exists() else {}

    evaluator = Evaluator(model=model, config=ckpt_cfg, device=dev_info.device)
    metrics, _ = evaluator.evaluate_dataset(
        dataset=dataset, annotations=annotations, output_dir=output_dir
    )

    print("=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    if metrics.is_synthetic or ckpt_meta.get("is_synthetic"):
        print("[WARNING] SYNTHETIC SMOKE-TEST DATA/CHECKPOINT — NOT REAL BENCHMARK METRICS")
    print(f"Videos Evaluated : {metrics.num_videos}")
    print(f"Frames Evaluated : {metrics.num_frames}")
    print(f"Clips Evaluated  : {metrics.num_clips}")
    print(f"Threshold        : {metrics.threshold:.2f}")
    print(f"Frame ROC-AUC    : {metrics.frame_roc_auc if metrics.frame_roc_auc is not None else 'N/A'}")
    print(f"Frame PR-AUC     : {metrics.frame_pr_auc if metrics.frame_pr_auc is not None else 'N/A'}")
    print(f"Frame Precision  : {metrics.frame_precision:.4f}")
    print(f"Frame Recall     : {metrics.frame_recall:.4f}")
    print(f"Frame F1 Score   : {metrics.frame_f1:.4f}")
    print(f"Clip ROC-AUC     : {metrics.clip_roc_auc if metrics.clip_roc_auc is not None else 'N/A'}")
    print(f"Video ROC-AUC    : {metrics.video_roc_auc if metrics.video_roc_auc is not None else 'N/A'}")
    print(f"Confusion Matrix : {metrics.frame_confusion_matrix}")
    if metrics.notes:
        print(f"Notes            : {metrics.notes}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
