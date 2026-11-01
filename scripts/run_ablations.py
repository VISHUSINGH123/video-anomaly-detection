"""Automated Ablation Study Suite for Video Anomaly Detection.

Trains and evaluates all 5 architectural variants under identical seeds, splits, and configurations:
- Variant A: VideoMAE-only Baseline (alpha = [1, 0, 0, 0])
- Variant B: DINOv2-only Baseline (alpha = [0, 1, 0, 0])
- Variant C: Concat-MLP (Ablation without Softmax Gating)
- Variant D: Dual-Stream Fusion without TCN (TCN disabled, alpha_t = 0)
- Variant E: Full Multi-Prior Gated Fusion (VideoMAE + DINOv2 + TCN + Motion Dynamics + Softmax Gating)

Aggregates metrics, logs gate allocations, exports CSV/JSON summaries, plots comparison charts,
and copies the top checkpoint to checkpoints/best.pt.
"""

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vad.config import load_config
from vad.data.annotations import load_annotations_directory
from vad.data.dataset import CachedFeatureDataset
from vad.device import resolve_device
from vad.evaluation.evaluator import Evaluator
from vad.models.anomaly_detector import AnomalyDetector
from vad.training.trainer import Trainer
from vad.utils.logging import setup_logger
from vad.utils.reproducibility import set_seed

logger = setup_logger("vad.run_ablations")

VARIANTS = [
    {
        "id": "exp_A_videomae",
        "name": "Variant A: VideoMAE-only Baseline",
        "fusion_mode": "videomae_only",
        "description": "VideoMAE spatiotemporal features only (alpha = [1, 0, 0, 0])",
    },
    {
        "id": "exp_B_dinov2",
        "name": "Variant B: DINOv2-only Baseline",
        "fusion_mode": "dinov2_only",
        "description": "DINOv2 appearance features only (alpha = [0, 1, 0, 0])",
    },
    {
        "id": "exp_C_concat_mlp",
        "name": "Variant C: Concat-MLP (No Gating)",
        "fusion_mode": "concat_mlp",
        "description": "Concat all streams into MLP projection without adaptive gating",
    },
    {
        "id": "exp_D_no_tcn",
        "name": "Variant D: Dual-Stream (No TCN)",
        "fusion_mode": "no_tcn",
        "description": "VideoMAE + DINOv2 + Motion without temporal convolution (alpha_t = 0)",
    },
    {
        "id": "exp_E_full",
        "name": "Variant E: Full Multi-Prior Gated Fusion",
        "fusion_mode": "gated_multiprior",
        "description": "Full VideoMAE + DINOv2 + TCN + Motion Dynamics + Temperature-scaled Softmax Gating",
    },
]


def plot_ablation_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    """Generate a publication-grade bar chart comparing metrics across ablation variants."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    x = np.arange(len(results_df))
    width = 0.25

    # Plot 1: Detection Performance Metrics
    r1 = ax1.bar(x - width, results_df["frame_roc_auc"] * 100, width, label="Frame ROC-AUC (%)", color="#1f77b4")
    r2 = ax1.bar(x, results_df["frame_pr_auc"] * 100, width, label="Frame PR-AUC (%)", color="#ff7f0e")
    r3 = ax1.bar(x + width, results_df["video_roc_auc"] * 100, width, label="Video ROC-AUC (%)", color="#2ca02c")

    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Detection Performance Across Architectural Variants", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    labels = [f"Variant {chr(65+i)}\n{row['fusion_mode']}" for i, (_, row) in enumerate(results_df.iterrows())]
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", framealpha=0.9)

    for bars in (r1, r2, r3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    # Plot 2: Compute Efficiency (Train Time vs Inference Latency)
    color_time = "#8c564b"
    color_lat = "#9467bd"
    ax2.bar(x - width/2, results_df["train_time_sec"], width, label="Train Time (s)", color=color_time)
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x + width/2, results_df["inference_latency_ms"], width, label="Inference Latency (ms/clip)", color=color_lat)

    ax2.set_ylabel("Training Time (seconds)", color=color_time, fontsize=11, fontweight="bold")
    ax2_twin.set_ylabel("Latency per Clip (ms)", color=color_lat, fontsize=11, fontweight="bold")
    ax2.set_title("Computational Footprint & Latency", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved ablation comparison plot to %s", output_path)


def run_ablation_suite(
    config_path: str = "configs/cpu.yaml",
    feature_dir: str = "data/processed/synthetic",
    annotations_dir: str = "data/annotations/synthetic",
    epochs: int = 8,
    output_dir: str = "outputs/evaluation/ablations",
    device_override: str | None = None,
) -> pd.DataFrame:
    """Execute all 5 ablation experiments sequentially, evaluate them, and log results."""
    base_cfg = load_config(config_path)
    base_cfg.training.epochs = epochs
    base_cfg.data.processed_feature_dir = feature_dir
    base_cfg.data.annotations_dir = annotations_dir
    if device_override:
        base_cfg.device = device_override

    dev_info = resolve_device(base_cfg.device)
    device = dev_info.device

    feat_path = Path(feature_dir)
    train_ds = CachedFeatureDataset(feature_dir=feat_path, split="train")
    val_ds = CachedFeatureDataset(feature_dir=feat_path, split="val")
    test_ds = CachedFeatureDataset(feature_dir=feat_path, split="test")

    ann_path = Path(annotations_dir)
    annotations = load_annotations_directory(ann_path) if ann_path.exists() else {}

    results: List[Dict[str, Any]] = []
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    best_overall_score = -1.0
    best_variant_ckpt: Path | None = None

    for variant in VARIANTS:
        v_id = variant["id"]
        v_name = variant["name"]
        v_mode = variant["fusion_mode"]
        logger.info("=" * 70)
        logger.info("RUNNING ABLATION: %s (%s)", v_name, v_mode)
        logger.info("=" * 70)

        cfg = copy.deepcopy(base_cfg)
        cfg.fusion.mode = v_mode
        cfg.training.checkpoint_dir = f"checkpoints/{v_id}"
        set_seed(cfg.seed)

        # 1. Train model
        model = AnomalyDetector(config=cfg)
        trainer = Trainer(model=model, config=cfg, device=device)
        t_start = time.perf_counter()
        train_summary = trainer.fit(train_dataset=train_ds, val_dataset=val_ds)
        t_train = time.perf_counter() - t_start

        best_ckpt = Path(train_summary["best_checkpoint"])

        # 2. Evaluate model on test set
        evaluator = Evaluator(model=trainer.model, config=cfg, device=device)
        eval_out_dir = out_path / v_id
        metrics, per_video_df = evaluator.evaluate_dataset(
            dataset=test_ds,
            annotations=annotations,
            output_dir=str(eval_out_dir),
        )

        # 3. Benchmark inference latency
        trainer.model.eval()
        dummy_v = torch.randn(1, 16, cfg.encoders.videomae.feature_dim, device=device)
        dummy_d = torch.randn(1, 16, cfg.encoders.dinov2.feature_dim, device=device)
        with torch.no_grad():
            for _ in range(5):
                _ = trainer.model(dummy_v, dummy_d)
            t_lat_start = time.perf_counter()
            n_iters = 30
            for _ in range(n_iters):
                _ = trainer.model(dummy_v, dummy_d)
            avg_latency_ms = ((time.perf_counter() - t_lat_start) / (n_iters * 16)) * 1000.0

        # 4. Extract gating weights if recorded
        gate_v = float(per_video_df["mean_gate_videomae"].mean()) if "mean_gate_videomae" in per_video_df else 0.0
        gate_d = float(per_video_df["mean_gate_dinov2"].mean()) if "mean_gate_dinov2" in per_video_df else 0.0
        gate_t = float(per_video_df["mean_gate_tcn"].mean()) if "mean_gate_tcn" in per_video_df else 0.0
        gate_m = float(per_video_df["mean_gate_motion"].mean()) if "mean_gate_motion" in per_video_df else 0.0

        score_for_rank = float(metrics.frame_roc_auc or 0.0) + float(metrics.frame_pr_auc or 0.0)
        if score_for_rank > best_overall_score:
            best_overall_score = score_for_rank
            best_variant_ckpt = best_ckpt

        record = {
            "variant_id": v_id,
            "variant_name": v_name,
            "fusion_mode": v_mode,
            "description": variant["description"],
            "frame_roc_auc": float(metrics.frame_roc_auc or 0.0),
            "frame_pr_auc": float(metrics.frame_pr_auc or 0.0),
            "frame_f1": float(metrics.frame_f1 or 0.0),
            "video_roc_auc": float(metrics.video_roc_auc or 0.0),
            "train_time_sec": round(t_train, 3),
            "inference_latency_ms": round(avg_latency_ms, 3),
            "gate_videomae": round(gate_v, 4),
            "gate_dinov2": round(gate_d, 4),
            "gate_tcn": round(gate_t, 4),
            "gate_motion": round(gate_m, 4),
            "checkpoint_path": str(best_ckpt),
        }
        results.append(record)

    results_df = pd.DataFrame(results)

    # Save summary CSV & JSON
    csv_path = out_path / "ablation_summary.csv"
    json_path = out_path / "ablation_summary.json"
    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Ablation summary saved to %s and %s", csv_path, json_path)

    # Plot visual comparison
    plot_path = out_path / "ablation_comparison.png"
    plot_ablation_comparison(results_df, plot_path)

    # Ensure best checkpoint is copied to checkpoints/best.pt
    if best_variant_ckpt and best_variant_ckpt.exists():
        best_pt_target = PROJECT_ROOT / "checkpoints/best.pt"
        shutil.copy2(best_variant_ckpt, best_pt_target)
        logger.info("Copied top performing checkpoint (%s) to %s", best_variant_ckpt, best_pt_target)

    return results_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full ablation suite across all 5 model variants.")
    parser.add_argument("--config", type=str, default="configs/cpu.yaml")
    parser.add_argument("--feature-dir", type=str, default="data/processed/synthetic")
    parser.add_argument("--annotations-dir", type=str, default="data/annotations/synthetic")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation/ablations")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    results_df = run_ablation_suite(
        config_path=args.config,
        feature_dir=args.feature_dir,
        annotations_dir=args.annotations_dir,
        epochs=args.epochs,
        output_dir=args.output_dir,
        device_override=args.device,
    )

    print("\n" + "=" * 90)
    print("VIDEO ANOMALY DETECTION — MULTI-PRIOR FUSION ABLATION BENCHMARK RESULTS")
    print("=" * 90)
    cols = ["variant_name", "fusion_mode", "frame_roc_auc", "frame_pr_auc", "video_roc_auc", "train_time_sec", "inference_latency_ms"]
    print(results_df[cols].to_string(index=False))
    print("=" * 90 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
