"""Configurable training loop with validation checkpoint selection, early stopping, and resume."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from vad.config import VADConfig
from vad.data.dataset import CachedFeatureDataset, collate_variable_length_features
from vad.evaluation.evaluator import Evaluator
from vad.models.anomaly_detector import AnomalyDetector
from vad.training.checkpoints import load_checkpoint, save_checkpoint
from vad.training.losses import AnomalyDetectionCriterion
from vad.utils.logging import setup_logger

logger = setup_logger("vad.trainer")


class Trainer:
    """Trains and validates the Multi-Prior Fusion AnomalyDetector on cached features."""

    def __init__(
        self,
        model: AnomalyDetector,
        config: VADConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self.criterion = AnomalyDetectionCriterion(config=self.config.training)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.config.training.epochs)
        )

    def train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch over `loader` and return average loss breakdown."""
        self.model.train()
        total_loss_accum = 0.0
        cls_loss_accum = 0.0
        smooth_loss_accum = 0.0
        sparse_loss_accum = 0.0
        num_batches = 0

        for batch in loader:
            v_feat = batch["videomae_features"].to(self.device)
            d_feat = batch["dinov2_features"].to(self.device)
            clip_labels = batch["clip_labels"].to(self.device)
            video_labels = batch["video_labels"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(v_feat, d_feat, mask=mask)
            loss_bd = self.criterion(
                logits=out.logits,
                scores=out.scores,
                clip_labels=clip_labels,
                video_labels=video_labels,
                mask=mask,
            )

            if not torch.isfinite(loss_bd.total_loss):
                raise RuntimeError(
                    f"Non-finite training loss encountered: {loss_bd.total_loss.item()}"
                )

            loss_bd.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss_accum += float(loss_bd.total_loss.item())
            cls_loss_accum += float(loss_bd.classification_loss.item())
            smooth_loss_accum += float(loss_bd.smoothness_loss.item())
            sparse_loss_accum += float(loss_bd.sparsity_loss.item())
            num_batches += 1

        denom = max(1, num_batches)
        return {
            "total_loss": total_loss_accum / denom,
            "classification_loss": cls_loss_accum / denom,
            "smoothness_loss": smooth_loss_accum / denom,
            "sparsity_loss": sparse_loss_accum / denom,
        }

    @torch.inference_mode()
    def compute_validation_loss(self, loader: DataLoader) -> float:
        """Compute average validation loss across `loader`."""
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in loader:
            v_feat = batch["videomae_features"].to(self.device)
            d_feat = batch["dinov2_features"].to(self.device)
            clip_labels = batch["clip_labels"].to(self.device)
            video_labels = batch["video_labels"].to(self.device)
            mask = batch["mask"].to(self.device)

            out = self.model(v_feat, d_feat, mask=mask)
            loss_bd = self.criterion(
                logits=out.logits,
                scores=out.scores,
                clip_labels=clip_labels,
                video_labels=video_labels,
                mask=mask,
            )
            total_loss += float(loss_bd.total_loss.item())
            n += 1
        return total_loss / max(1, n)

    def fit(
        self,
        train_dataset: CachedFeatureDataset,
        val_dataset: Optional[CachedFeatureDataset] = None,
        resume_from: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Train the model, select best checkpoint on validation metrics/loss, and log history."""
        start_epoch = 1
        if resume_from is not None:
            ckpt_model, _, meta = load_checkpoint(
                resume_from, device=self.device, optimizer=self.optimizer
            )
            self.model.load_state_dict(ckpt_model.state_dict())
            start_epoch = int(meta.get("epoch", 0)) + 1
            logger.info("Resumed training from %s at epoch %d", resume_from, start_epoch)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate_variable_length_features,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                collate_fn=collate_variable_length_features,
            )
            if val_dataset is not None
            else None
        )

        is_synthetic = any(b.is_synthetic for b in train_dataset.bundles)
        if val_dataset is not None:
            is_synthetic = is_synthetic or any(b.is_synthetic for b in val_dataset.bundles)

        ckpt_dir = Path(self.config.training.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = ckpt_dir / "best.pt"
        last_ckpt_path = ckpt_dir / "last.pt"

        best_score = -float("inf")
        patience_counter = 0
        history: List[Dict[str, Any]] = []
        start_time = time.perf_counter()

        for epoch in range(start_epoch, self.config.training.epochs + 1):
            t0 = time.perf_counter()
            train_stats = self.train_one_epoch(train_loader)
            self.scheduler.step()

            epoch_record: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(train_stats["total_loss"], 6),
                "train_cls_loss": round(train_stats["classification_loss"], 6),
                "train_smooth_loss": round(train_stats["smoothness_loss"], 6),
                "train_sparse_loss": round(train_stats["sparsity_loss"], 6),
                "lr": round(float(self.optimizer.param_groups[0]["lr"]), 8),
            }

            if val_dataset is not None and val_loader is not None:
                val_loss = self.compute_validation_loss(val_loader)
                evaluator = Evaluator(self.model, self.config, device=self.device)
                val_metrics, _ = evaluator.evaluate_dataset(val_dataset)
                epoch_record["val_loss"] = round(val_loss, 6)
                epoch_record["val_frame_roc_auc"] = val_metrics.frame_roc_auc
                epoch_record["val_frame_pr_auc"] = val_metrics.frame_pr_auc

                # Use Frame ROC-AUC when both classes exist in validation; otherwise use -val_loss
                selection_criterion = (
                    val_metrics.frame_roc_auc
                    if val_metrics.frame_roc_auc is not None
                    else -val_loss
                )
            else:
                selection_criterion = -train_stats["total_loss"]

            epoch_record["elapsed_sec"] = round(time.perf_counter() - t0, 3)
            history.append(epoch_record)

            save_checkpoint(
                last_ckpt_path,
                model=self.model,
                config=self.config,
                epoch=epoch,
                metrics=epoch_record,
                optimizer=self.optimizer,
                is_synthetic=is_synthetic,
            )

            if selection_criterion > best_score + 1e-6:
                best_score = selection_criterion
                patience_counter = 0
                save_checkpoint(
                    best_ckpt_path,
                    model=self.model,
                    config=self.config,
                    epoch=epoch,
                    metrics=epoch_record,
                    optimizer=self.optimizer,
                    is_synthetic=is_synthetic,
                )
                logger.info(
                    "Epoch %02d/%02d | train_loss=%.4f | criterion=%.4f [NEW BEST -> %s]",
                    epoch,
                    self.config.training.epochs,
                    train_stats["total_loss"],
                    selection_criterion,
                    best_ckpt_path.name,
                )
            else:
                patience_counter += 1
                logger.info(
                    "Epoch %02d/%02d | train_loss=%.4f | criterion=%.4f (patience %d/%d)",
                    epoch,
                    self.config.training.epochs,
                    train_stats["total_loss"],
                    selection_criterion,
                    patience_counter,
                    self.config.training.early_stopping_patience,
                )
                if patience_counter >= self.config.training.early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        total_duration = time.perf_counter() - start_time
        summary = {
            "experiment_name": self.config.experiment_name,
            "is_synthetic": is_synthetic,
            "epochs_completed": len(history),
            "best_criterion": round(float(best_score), 6),
            "total_training_sec": round(total_duration, 3),
            "best_checkpoint": str(best_ckpt_path),
            "last_checkpoint": str(last_ckpt_path),
            "history": history,
        }

        history_path = ckpt_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
