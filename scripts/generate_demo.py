"""End-to-end synthetic smoke-test pipeline generator.

Generates synthetic surveillance videos, extracts synthetic features, trains the TCN + Multi-Prior
Fusion model for a few epochs, evaluates on the synthetic test split, and produces a complete
prediction bundle (CSV, timeline PNG, and annotated MP4) in `outputs/demo/`.

All generated artifacts are explicitly marked as `is_synthetic=True` so they are never confused
with genuine benchmark evaluation.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.download_sample_data import generate_synthetic_dataset
from vad.config import load_config
from vad.data.annotations import load_annotations_directory
from vad.data.dataset import CachedFeatureDataset
from vad.device import resolve_device
from vad.evaluation.evaluator import Evaluator
from vad.inference.predictor import VideoAnomalyPredictor
from vad.inference.visualization import export_annotated_video, plot_anomaly_timeline
from vad.models.anomaly_detector import AnomalyDetector
from vad.training.trainer import Trainer
from vad.utils.logging import setup_logger
from vad.utils.reproducibility import set_seed

logger = setup_logger("vad.generate_demo")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run end-to-end synthetic smoke test: data -> train -> evaluate -> predict."
    )
    parser.add_argument("--config", type=str, default="configs/cpu.yaml")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs/demo")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg.training.epochs = args.epochs
    cfg.data.raw_video_dir = "data/raw/synthetic"
    cfg.data.processed_feature_dir = "data/processed/synthetic"
    cfg.data.annotations_dir = "data/annotations/synthetic"
    cfg.training.checkpoint_dir = "checkpoints/synthetic_demo"
    cfg.evaluation.output_dir = f"{args.output_dir}/evaluation"
    cfg.inference.output_dir = f"{args.output_dir}/predictions"
    set_seed(cfg.seed)

    dev_info = resolve_device(cfg.device)
    logger.info("Step 1/4: Generating synthetic smoke-test videos and cached features...")
    paths = generate_synthetic_dataset(
        raw_dir=Path(cfg.data.raw_video_dir),
        processed_dir=Path(cfg.data.processed_feature_dir),
        annotations_dir=Path(cfg.data.annotations_dir),
        clip_length=cfg.data.clip_length,
        clip_stride=cfg.data.clip_stride,
        feature_dim=cfg.encoders.videomae.feature_dim,
        seed=cfg.seed,
    )

    logger.info("Step 2/4: Running synthetic smoke training (%d epochs)...", cfg.training.epochs)
    train_ds = CachedFeatureDataset(paths["processed_dir"], split="train")
    val_ds = CachedFeatureDataset(paths["processed_dir"], split="val")
    test_ds = CachedFeatureDataset(paths["processed_dir"], split="test")

    model = AnomalyDetector(config=cfg)
    trainer = Trainer(model=model, config=cfg, device=dev_info.device)
    train_summary = trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    logger.info("Step 3/4: Evaluating best synthetic checkpoint on synthetic test split...")
    best_ckpt = Path(train_summary["best_checkpoint"])
    annotations = load_annotations_directory(paths["annotations_file"])
    evaluator = Evaluator(model=trainer.model, config=cfg, device=dev_info.device)
    metrics, _ = evaluator.evaluate_dataset(
        dataset=test_ds,
        annotations=annotations,
        output_dir=cfg.evaluation.output_dir,
    )
    logger.info(
        "Synthetic Test Frame ROC-AUC: %s | Frame PR-AUC: %s",
        metrics.frame_roc_auc,
        metrics.frame_pr_auc,
    )

    logger.info("Step 4/4: Running single-video sliding-window inference and exporting overlay...")
    predictor = VideoAnomalyPredictor.from_checkpoint(
        checkpoint_path=best_ckpt,
        device_override=cfg.device,
        use_synthetic_encoders=True,
    )
    sample_video = Path(cfg.data.raw_video_dir) / "synthetic_test_anom_01.mp4"
    pred_result, decoded = predictor.predict_video_file(sample_video)

    pred_dir = Path(cfg.inference.output_dir)
    clip_csv, frame_csv = pred_result.save_csv(pred_dir)
    fig = plot_anomaly_timeline(
        pred_result,
        output_path=pred_dir / f"{pred_result.video_id}_timeline.png",
        ground_truth_intervals=[(16, 47)],
    )
    plt.close(fig)
    overlay_mp4 = export_annotated_video(
        decoded.frames_rgb,
        pred_result,
        pred_dir / f"{pred_result.video_id}_annotated.mp4",
    )

    logger.info("Demo artifacts successfully generated in %s (clip CSV: %s, video: %s)", pred_dir, clip_csv, overlay_mp4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
