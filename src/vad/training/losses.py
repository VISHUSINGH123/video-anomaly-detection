"""Loss functions for supervised clip-level and weakly-supervised MIL video anomaly learning."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vad.config import TrainingConfig


@dataclass
class LossBreakdown:
    """Detailed loss components for logging and diagnostics."""

    total_loss: torch.Tensor
    classification_loss: torch.Tensor
    smoothness_loss: torch.Tensor
    sparsity_loss: torch.Tensor


def temporal_smoothness_penalty(
    scores: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute first-order temporal smoothness penalty `mean((s_t - s_{t-1})^2)` over valid adjacent clips."""
    if scores.shape[1] < 2:
        return torch.tensor(0.0, dtype=scores.dtype, device=scores.device)

    diffs = (scores[:, 1:] - scores[:, :-1]) ** 2
    if mask is not None:
        valid_pairs = (mask[:, 1:] & mask[:, :-1]).to(scores.dtype)
        denom = valid_pairs.sum().clamp(min=1.0)
        return (diffs * valid_pairs).sum() / denom
    return diffs.mean()


def temporal_sparsity_penalty(
    scores: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute L1 sparsity penalty on predicted anomaly scores so normal clips stay near zero."""
    if mask is not None:
        m = mask.to(scores.dtype)
        denom = m.sum().clamp(min=1.0)
        return (scores * m).sum() / denom
    return scores.mean()


class AnomalyDetectionCriterion(nn.Module):
    """Unified loss function supporting both frame/clip supervision and weakly-supervised Top-K MIL."""

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        super().__init__()
        self.config = config or TrainingConfig()

    def _supervised_clip_loss(
        self,
        logits: torch.Tensor,
        clip_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Weighted binary cross-entropy over valid temporal clips."""
        pos_w = torch.tensor(self.config.pos_weight, dtype=logits.dtype, device=logits.device)
        raw_bce = F.binary_cross_entropy_with_logits(
            logits, clip_labels, pos_weight=pos_w, reduction="none"
        )
        if mask is not None:
            m = mask.to(logits.dtype)
            denom = m.sum().clamp(min=1.0)
            return (raw_bce * m).sum() / denom
        return raw_bce.mean()

    def _weakly_supervised_mil_loss(
        self,
        logits: torch.Tensor,
        video_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Top-K Multiple-Instance Learning (MIL) bag loss over video-level labels.

        For each video `i` with `L_i` valid clips, aggregates the top `k = ceil(rho * L_i)` clip
        logits into a bag logit `z_bag(i) = (1/k) sum_{t in TopK} z_{i,t}` and applies weighted BCE
        against `video_labels[i] in {0, 1}`.
        """
        batch_size = logits.shape[0]
        bag_logits = []

        for i in range(batch_size):
            if mask is not None:
                valid_logits = logits[i][mask[i]]
            else:
                valid_logits = logits[i]

            if valid_logits.numel() == 0:
                bag_logits.append(torch.tensor(0.0, dtype=logits.dtype, device=logits.device))
                continue

            k = max(1, int(math.ceil(self.config.mil_topk_ratio * valid_logits.numel())))
            topk_vals, _ = torch.topk(valid_logits, k=k)
            bag_logits.append(topk_vals.mean())

        bag_logits_tensor = torch.stack(bag_logits, dim=0)
        pos_w = torch.tensor(self.config.pos_weight, dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(
            bag_logits_tensor, video_labels.float(), pos_weight=pos_w
        )

    def forward(
        self,
        logits: torch.Tensor,
        scores: torch.Tensor,
        clip_labels: torch.Tensor,
        video_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> LossBreakdown:
        """Compute total training loss and its constituent components."""
        if self.config.supervision == "supervised_clip":
            cls_loss = self._supervised_clip_loss(logits, clip_labels, mask=mask)
        elif self.config.supervision == "weakly_supervised_mil":
            cls_loss = self._weakly_supervised_mil_loss(logits, video_labels, mask=mask)
        else:
            raise ValueError(f"Unsupported supervision mode: {self.config.supervision}")

        smooth_loss = temporal_smoothness_penalty(scores, mask=mask)
        sparse_loss = temporal_sparsity_penalty(scores, mask=mask)

        total = (
            cls_loss
            + self.config.smoothness_lambda * smooth_loss
            + self.config.sparsity_lambda * sparse_loss
        )
        return LossBreakdown(
            total_loss=total,
            classification_loss=cls_loss,
            smoothness_loss=smooth_loss,
            sparsity_loss=sparse_loss,
        )
