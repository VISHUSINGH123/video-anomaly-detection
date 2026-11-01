"""CLI script to train the Temporal Convolutional Network and Multi-Prior Fusion Anomaly Detector."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vad.config import load_config
from vad.data.dataset import CachedFeatureDataset
from vad.device import resolve_device
from vad.models.anomaly_detector import AnomalyDetector
from vad.training.trainer import Trainer
from vad.utils.logging import setup_logger
from vad.utils.reproducibility import set_seed

logger = setup_logger("vad.train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Video Anomaly Detection TCN + Multi-Prior Fusion model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=None,
        help="Directory containing cached .npz feature files (overrides config.data.processed_feature_dir).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints (overrides config.training.checkpoint_dir).",
    )
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default=None,
        choices=["gated_multiprior", "concat_mlp", "videomae_only", "dinov2_only", "no_tcn"],
        help="Optional ablation mode override for multi-prior fusion.",
    )
    parser.add_argument(
        "--supervision",
        type=str,
        default=None,
        choices=["supervised_clip", "weakly_supervised_mil"],
        help="Supervision strategy override.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device override (auto, cpu, cuda, directml, mps).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.feature_dir:
        cfg.data.processed_feature_dir = args.feature_dir
    if args.checkpoint_dir:
        cfg.training.checkpoint_dir = args.checkpoint_dir
    if args.fusion_mode:
        cfg.fusion.mode = args.fusion_mode
    if args.supervision:
        cfg.training.supervision = args.supervision
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.device:
        cfg.device = args.device
    cfg.validate()

    set_seed(cfg.seed)
    dev_info = resolve_device(cfg.device)
    logger.info(
        "Starting training | Experiment: %s | Mode: %s | Supervision: %s | Device: %s",
        cfg.experiment_name,
        cfg.fusion.mode,
        cfg.training.supervision,
        dev_info.device_name,
    )

    feature_dir = Path(cfg.data.processed_feature_dir)
    train_ds = CachedFeatureDataset(
        feature_dir=feature_dir,
        split="train",
        videomae_dim=cfg.encoders.videomae.feature_dim,
        dinov2_dim=cfg.encoders.dinov2.feature_dim,
    )

    try:
        val_ds: CachedFeatureDataset | None = CachedFeatureDataset(
            feature_dir=feature_dir,
            split="val",
            videomae_dim=cfg.encoders.videomae.feature_dim,
            dinov2_dim=cfg.encoders.dinov2.feature_dim,
        )
    except ValueError:
        try:
            val_ds = CachedFeatureDataset(
                feature_dir=feature_dir,
                split="test",
                videomae_dim=cfg.encoders.videomae.feature_dim,
                dinov2_dim=cfg.encoders.dinov2.feature_dim,
            )
        except ValueError:
            val_ds = None

    if any(b.is_synthetic for b in train_ds.bundles):
        logger.warning(
            "SYNTHETIC SMOKE-TRAINING MODE ACTIVE: Training on synthetic feature bundles. "
            "Checkpoints will be marked is_synthetic=True and must not be reported as benchmark results."
        )

    model = AnomalyDetector(config=cfg)
    trainer = Trainer(model=model, config=cfg, device=dev_info.device)
    summary = trainer.fit(train_dataset=train_ds, val_dataset=val_ds, resume_from=args.resume)

    logger.info(
        "Training complete in %.2fs | Best checkpoint saved to: %s",
        summary["total_training_sec"],
        summary["best_checkpoint"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
