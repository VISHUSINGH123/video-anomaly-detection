"""End-to-end tests for annotation alignment, feature caching, checkpointing, training, inference, and CLI."""

from pathlib import Path

import pytest
import torch

from scripts.download_sample_data import generate_synthetic_dataset
from scripts.evaluate import build_parser as build_eval_parser
from scripts.extract_features import build_parser as build_extract_parser
from scripts.predict_video import build_parser as build_predict_parser
from scripts.train import build_parser as build_train_parser
from vad.config import VADConfig
from vad.data.annotations import (
    VideoAnnotation,
    align_annotations_to_windows,
    parse_ucsd_matlab_annotations,
)
from vad.data.dataset import CachedFeatureDataset, load_cached_features
from vad.evaluation.evaluator import Evaluator
from vad.inference.predictor import VideoAnomalyPredictor
from vad.models.anomaly_detector import AnomalyDetector
from vad.training.checkpoints import load_checkpoint, save_checkpoint
from vad.training.trainer import Trainer


def test_ucsd_matlab_annotation_parser_and_alignment(tmp_path: Path) -> None:
    m_file = tmp_path / "UCSDped2.m"
    m_file.write_text(
        "TestVideoFile{end+1}.gt_frame = [61:180];\n"
        "TestVideoFile{end+1}.gt_frame = [95:180];\n",
        encoding="utf-8",
    )
    parsed = parse_ucsd_matlab_annotations(m_file)
    assert "Test001" in parsed and "Test002" in parsed
    assert parsed["Test001"].anomaly_intervals == [(60, 179)]
    assert parsed["Test001"].video_label == 1

    frame_gt = parsed["Test001"].get_frame_labels(180)
    assert frame_gt[:60].sum() == 0
    assert frame_gt[60:180].sum() == 120


def test_checkpoint_save_and_load_roundtrip(tmp_path: Path) -> None:
    cfg = VADConfig()
    cfg.encoders.videomae.feature_dim = 64
    cfg.encoders.dinov2.feature_dim = 64
    cfg.fusion.proj_dim = 32
    cfg.temporal.hidden_dim = 32

    model = AnomalyDetector(config=cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_file = tmp_path / "model_ckpt.pt"
    save_checkpoint(
        ckpt_file,
        model=model,
        config=cfg,
        epoch=5,
        metrics={"val_frame_roc_auc": 0.91},
        optimizer=optimizer,
        is_synthetic=True,
    )

    loaded_model, loaded_cfg, meta = load_checkpoint(ckpt_file)
    assert meta["epoch"] == 5
    assert meta["is_synthetic"] is True
    assert loaded_cfg.fusion.proj_dim == 32

    x_v = torch.randn(1, 6, 64)
    x_d = torch.randn(1, 6, 64)
    model.eval()
    loaded_model.eval()
    with torch.no_grad():
        s1 = model(x_v, x_d).scores
        s2 = loaded_model(x_v, x_d).scores
    assert torch.allclose(s1, s2, atol=1e-6)


def test_end_to_end_synthetic_smoke_pipeline(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    ann_dir = tmp_path / "annotations"
    ckpt_dir = tmp_path / "checkpoints"
    eval_dir = tmp_path / "eval"
    pred_dir = tmp_path / "pred"

    paths = generate_synthetic_dataset(
        raw_dir=raw_dir,
        processed_dir=proc_dir,
        annotations_dir=ann_dir,
        clip_length=16,
        clip_stride=16,
        feature_dim=64,
        seed=42,
    )

    sample_bundle = load_cached_features(proc_dir / "synthetic_train_anom_01.npz")
    assert sample_bundle.is_synthetic is True
    assert sample_bundle.videomae_features is not None
    assert sample_bundle.videomae_features.shape == (4, 64)

    cfg = VADConfig()
    cfg.device = "cpu"
    cfg.encoders.videomae.feature_dim = 64
    cfg.encoders.dinov2.feature_dim = 64
    cfg.temporal.hidden_dim = 32
    cfg.temporal.num_layers = 2
    cfg.fusion.proj_dim = 32
    cfg.training.epochs = 3
    cfg.training.batch_size = 2
    cfg.training.checkpoint_dir = str(ckpt_dir)

    train_ds = CachedFeatureDataset(proc_dir, split="train", videomae_dim=64, dinov2_dim=64)
    val_ds = CachedFeatureDataset(proc_dir, split="val", videomae_dim=64, dinov2_dim=64)
    test_ds = CachedFeatureDataset(proc_dir, split="test", videomae_dim=64, dinov2_dim=64)

    detector = AnomalyDetector(config=cfg)
    trainer = Trainer(model=detector, config=cfg, device=torch.device("cpu"))
    summary = trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    best_ckpt = Path(summary["best_checkpoint"])
    assert best_ckpt.exists()
    assert summary["is_synthetic"] is True

    # Evaluate on test split
    evaluator = Evaluator(model=trainer.model, config=cfg, device=torch.device("cpu"))
    metrics, details = evaluator.evaluate_dataset(test_ds, output_dir=eval_dir)
    assert metrics.is_synthetic is True
    assert (eval_dir / "evaluation_summary.json").exists()
    assert (eval_dir / "per_video_summary.csv").exists()

    # Weakly supervised MIL smoke check
    cfg.training.supervision = "weakly_supervised_mil"
    cfg.training.epochs = 1
    mil_trainer = Trainer(model=detector, config=cfg, device=torch.device("cpu"))
    mil_summary = mil_trainer.fit(train_dataset=train_ds)
    assert mil_summary["epochs_completed"] == 1

    # Single-video inference smoke check
    predictor = VideoAnomalyPredictor.from_checkpoint(
        checkpoint_path=best_ckpt,
        device_override="cpu",
        use_synthetic_encoders=True,
    )
    vid_file = raw_dir / "synthetic_test_anom_01.mp4"
    pred_res, _ = predictor.predict_video_file(vid_file, window_length=16, window_stride=16)
    assert pred_res.is_synthetic_mode is True
    assert pred_res.frame_scores.shape == (64,)
    clip_csv, frame_csv = pred_res.save_csv(pred_dir)
    assert clip_csv.exists() and frame_csv.exists()


def test_cli_argument_parsers_validation() -> None:
    extract_p = build_extract_parser()
    ext_args = extract_p.parse_args(["--config", "configs/default.yaml", "--video-dir", "data/raw"])
    assert ext_args.config == "configs/default.yaml"
    assert ext_args.video_dir == "data/raw"

    train_p = build_train_parser()
    tr_args = train_p.parse_args(["--config", "configs/cpu.yaml", "--fusion-mode", "no_tcn"])
    assert tr_args.fusion_mode == "no_tcn"

    eval_p = build_eval_parser()
    ev_args = eval_p.parse_args(["--config", "configs/default.yaml", "--checkpoint", "checkpoints/best.pt"])
    assert ev_args.checkpoint == "checkpoints/best.pt"

    pred_p = build_predict_parser()
    pr_args = pred_p.parse_args(["--video", "sample.mp4", "--checkpoint", "checkpoints/best.pt"])
    assert pr_args.video == "sample.mp4"

    # Invalid fusion mode must fail argument validation
    with pytest.raises(SystemExit):
        train_p.parse_args(["--fusion-mode", "invalid_mode"])
