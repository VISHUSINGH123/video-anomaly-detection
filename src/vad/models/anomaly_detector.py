"""End-to-end Temporal Anomaly Detector combining Multi-Prior Fusion and Scoring Head."""

from typing import Optional

import torch
import torch.nn as nn

from vad.config import VADConfig
from vad.models.fusion import FusionOutput, MultiPriorFusionModule


class AnomalyDetector(nn.Module):
    """Trainable Temporal Anomaly Detector operating on aligned VideoMAE and DINOv2 clip features."""

    def __init__(self, config: Optional[VADConfig] = None) -> None:
        super().__init__()
        self.config = config or VADConfig()
        d_proj = self.config.fusion.proj_dim
        drop = self.config.fusion.dropout

        self.fusion = MultiPriorFusionModule(
            videomae_dim=self.config.encoders.videomae.feature_dim,
            dinov2_dim=self.config.encoders.dinov2.feature_dim,
            temporal_cfg=self.config.temporal,
            fusion_cfg=self.config.fusion,
        )

        self.scoring_head = nn.Sequential(
            nn.Linear(d_proj, d_proj // 2),
            nn.LayerNorm(d_proj // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_proj // 2, 1),
        )

    def forward(
        self,
        videomae_features: torch.Tensor,
        dinov2_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> FusionOutput:
        """Compute per-clip anomaly probabilities and inspectable prior gate weights.

        Args:
            videomae_features: Float tensor `(B, T, D_v)`.
            dinov2_features: Float tensor `(B, T, D_d)`.
            mask: Optional boolean validity mask `(B, T)` where True = valid clip.

        Returns:
            FusionOutput with `scores` in `(0, 1)`, raw `logits`, `fused_features`,
            `gate_weights`, `prior_names`, and `motion_magnitude`.
        """
        if not self.config.encoders.videomae.enabled:
            videomae_features = torch.zeros_like(videomae_features)
        if not self.config.encoders.dinov2.enabled:
            dinov2_features = torch.zeros_like(dinov2_features)

        fused, gate_weights, motion_mag = self.fusion(
            videomae_features=videomae_features,
            dinov2_features=dinov2_features,
            mask=mask,
        )

        logits = self.scoring_head(fused).squeeze(-1)  # (B, T)
        scores = torch.sigmoid(logits)

        if mask is not None:
            scores = scores * mask.to(scores.dtype)
            logits = logits * mask.to(logits.dtype)

        return FusionOutput(
            scores=scores,
            logits=logits,
            fused_features=fused,
            gate_weights=gate_weights,
            prior_names=self.fusion.prior_names,
            motion_magnitude=motion_mag,
        )
