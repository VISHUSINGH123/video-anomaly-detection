"""Multi-Prior Feature Fusion and Anomaly Scoring Module.

Combines four complementary priors for video anomaly detection:
1. Spatiotemporal semantic prior (`h_v` from VideoMAE).
2. Frame-level appearance prior (`h_d` from DINOv2).
3. Multi-scale dilated temporal context prior (`h_t` from TCN).
4. Temporal feature-change (motion/novelty velocity) prior (`h_m`), derived from first-order
   temporal feature differences `Delta h_v(t) = h_v(t) - h_v(t-1)` and `Delta h_d(t) = h_d(t) - h_d(t-1)`.

A learned temperature-scaled softmax gating network predicts per-clip normalized prior weights
`alpha(t) = [alpha_v(t), alpha_d(t), alpha_t(t), alpha_m(t)]` satisfying `sum_k alpha_k(t) = 1`.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vad.config import FusionConfig, TemporalConfig
from vad.models.temporal_encoder import TemporalConvNet


@dataclass
class FusionOutput:
    """Structured output of the Multi-Prior Fusion and Anomaly Detector."""

    scores: torch.Tensor  # Continuous anomaly scores in (0, 1), shape (B, T)
    logits: torch.Tensor  # Raw pre-sigmoid anomaly logits, shape (B, T)
    fused_features: torch.Tensor  # Fused clip representations, shape (B, T, proj_dim)
    gate_weights: torch.Tensor  # Inspectable normalized prior weights, shape (B, T, num_priors)
    prior_names: tuple[str, ...]  # Names corresponding to the last dimension of gate_weights
    motion_magnitude: torch.Tensor  # L2 feature change magnitude per clip, shape (B, T)


class StreamProjector(nn.Module):
    """L2-normalizing linear projection with LayerNorm, GELU, and Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2-normalize raw foundation model embeddings along feature axis for scale stability
        x_norm = F.normalize(x, p=2.0, dim=-1, eps=1e-6)
        return self.drop(self.act(self.norm(self.proj(x_norm))))


class MultiPriorFusionModule(nn.Module):
    """Trainable multi-prior feature fusion with inspectable gating and ablation modes."""

    def __init__(
        self,
        videomae_dim: int = 768,
        dinov2_dim: int = 768,
        temporal_cfg: Optional[TemporalConfig] = None,
        fusion_cfg: Optional[FusionConfig] = None,
    ) -> None:
        super().__init__()
        self.temporal_cfg = temporal_cfg or TemporalConfig()
        self.fusion_cfg = fusion_cfg or FusionConfig()

        d_proj = self.fusion_cfg.proj_dim
        drop = self.fusion_cfg.dropout

        # 1. Spatiotemporal stream projector (VideoMAE)
        self.videomae_proj = StreamProjector(videomae_dim, d_proj, dropout=drop)

        # 2. Appearance stream projector (DINOv2)
        self.dinov2_proj = StreamProjector(dinov2_dim, d_proj, dropout=drop)

        # Joint projection into TCN
        self.tcn_in_proj = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj),
            nn.LayerNorm(d_proj),
            nn.GELU(),
        )

        # 3. Temporal Convolutional Network stream
        self.tcn = TemporalConvNet(
            in_dim=d_proj,
            hidden_dim=self.temporal_cfg.hidden_dim,
            num_layers=self.temporal_cfg.num_layers,
            kernel_size=self.temporal_cfg.kernel_size,
            dilation_base=self.temporal_cfg.dilation_base,
            dropout=self.temporal_cfg.dropout,
            causal=self.temporal_cfg.causal,
        )
        self.tcn_out_proj = (
            nn.Linear(self.temporal_cfg.hidden_dim, d_proj)
            if self.temporal_cfg.hidden_dim != d_proj
            else nn.Identity()
        )

        # 4. Temporal feature-change (motion/novelty velocity) prior
        self.motion_proj = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj),
            nn.LayerNorm(d_proj),
            nn.GELU(),
            nn.Dropout(drop),
        )

        self.prior_names = ("videomae", "dinov2", "tcn_temporal", "motion_dynamics")
        num_priors = len(self.prior_names)

        # Adaptive gating network over the 4 priors (registered when mode != "concat_mlp")
        if self.fusion_cfg.mode == "concat_mlp":
            self.gate_network: Optional[nn.Module] = None
            self.concat_mlp: Optional[nn.Module] = nn.Sequential(
                nn.Linear(d_proj * num_priors, d_proj),
                nn.LayerNorm(d_proj),
                nn.GELU(),
                nn.Dropout(drop),
            )
        else:
            self.gate_network = nn.Sequential(
                nn.Linear(d_proj * num_priors, d_proj),
                nn.LayerNorm(d_proj),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(d_proj, num_priors),
            )
            self.concat_mlp = None

        self.final_norm = nn.LayerNorm(d_proj)

    @staticmethod
    def _compute_temporal_difference(
        h: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute causal first-order temporal difference `Delta h(t) = h(t) - h(t-1)`."""
        diff = torch.zeros_like(h)
        if h.shape[1] > 1:
            diff[:, 1:, :] = h[:, 1:, :] - h[:, :-1, :]
        if mask is not None:
            diff = diff * mask.unsqueeze(-1).to(diff.dtype)
        return diff

    def _build_ablation_mask(self, device: torch.device) -> torch.Tensor:
        """Return a binary mask `(4,)` indicating which priors are active for the configured mode."""
        mode = self.fusion_cfg.mode
        use_tcn = self.temporal_cfg.enabled and (mode not in ("videomae_only", "dinov2_only", "no_tcn"))
        use_motion = self.fusion_cfg.include_motion_prior and (mode not in ("videomae_only", "dinov2_only"))

        if mode == "videomae_only":
            active = [1.0, 0.0, 0.0, 0.0]
        elif mode == "dinov2_only":
            active = [0.0, 1.0, 0.0, 0.0]
        elif mode == "no_tcn":
            active = [1.0, 1.0, 0.0, 1.0 if use_motion else 0.0]
        else:
            active = [
                1.0,
                1.0,
                1.0 if use_tcn else 0.0,
                1.0 if use_motion else 0.0,
            ]
        return torch.tensor(active, dtype=torch.float32, device=device)

    def forward(
        self,
        videomae_features: torch.Tensor,
        dinov2_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse multi-stream features across time.

        Args:
            videomae_features: Float tensor `(B, T, D_v)`.
            dinov2_features: Float tensor `(B, T, D_d)`.
            mask: Optional boolean mask `(B, T)`.

        Returns:
            Tuple of:
              - fused_features: `(B, T, proj_dim)`
              - gate_weights: `(B, T, 4)` normalized weights across priors
              - motion_magnitude: `(B, T)` L2 norm of temporal feature velocity
        """
        if videomae_features.ndim != 3 or dinov2_features.ndim != 3:
            raise ValueError(
                f"Expected 3D tensors (B, T, D), got videomae={tuple(videomae_features.shape)}, "
                f"dinov2={tuple(dinov2_features.shape)}"
            )
        if videomae_features.shape[:2] != dinov2_features.shape[:2]:
            raise ValueError(
                f"Batch/time dimension mismatch between VideoMAE {tuple(videomae_features.shape[:2])} "
                f"and DINOv2 {tuple(dinov2_features.shape[:2])}"
            )

        mode = self.fusion_cfg.mode
        h_v = self.videomae_proj(videomae_features)
        h_d = self.dinov2_proj(dinov2_features)

        if mode == "videomae_only":
            h_d = torch.zeros_like(h_d)
        elif mode == "dinov2_only":
            h_v = torch.zeros_like(h_v)

        # Temporal differences (motion/novelty velocity prior)
        diff_v = self._compute_temporal_difference(h_v, mask=mask)
        diff_d = self._compute_temporal_difference(h_d, mask=mask)
        diff_concat = torch.cat([diff_v, diff_d], dim=-1)
        motion_mag = torch.linalg.vector_norm(diff_concat, ord=2, dim=-1)  # (B, T)
        h_m = self.motion_proj(diff_concat)

        # TCN temporal context prior
        joint_in = self.tcn_in_proj(torch.cat([h_v, h_d], dim=-1))
        if self.temporal_cfg.enabled and mode not in ("videomae_only", "dinov2_only", "no_tcn"):
            h_t = self.tcn_out_proj(self.tcn(joint_in, mask=mask))
        else:
            h_t = torch.zeros_like(h_v)

        active_mask = self._build_ablation_mask(h_v.device)  # (4,)
        priors_stack = torch.stack([h_v, h_d, h_t, h_m], dim=-2)  # (B, T, 4, D_proj)
        priors_stack = priors_stack * active_mask.view(1, 1, 4, 1)

        concat_all = priors_stack.reshape(h_v.shape[0], h_v.shape[1], -1)  # (B, T, 4 * D_proj)

        if mode == "concat_mlp":
            assert self.concat_mlp is not None
            fused = self.concat_mlp(concat_all)
            # Uniform weights over active priors for inspection
            w = active_mask / active_mask.sum().clamp(min=1.0)
            gate_weights = w.view(1, 1, 4).expand(h_v.shape[0], h_v.shape[1], 4)
        else:
            assert self.gate_network is not None
            raw_gate_logits = self.gate_network(concat_all) / self.fusion_cfg.temperature
            # Mask out disabled priors with -1e9 before softmax
            inactive_bool = active_mask < 0.5
            if inactive_bool.any():
                raw_gate_logits = raw_gate_logits.masked_fill(
                    inactive_bool.view(1, 1, 4), -1e9
                )
            gate_weights = F.softmax(raw_gate_logits, dim=-1)  # (B, T, 4)
            fused = torch.sum(priors_stack * gate_weights.unsqueeze(-1), dim=-2)

        fused = self.final_norm(fused)
        if mask is not None:
            m_float = mask.unsqueeze(-1).to(fused.dtype)
            fused = fused * m_float
            motion_mag = motion_mag * mask.to(motion_mag.dtype)

        return fused, gate_weights, motion_mag
