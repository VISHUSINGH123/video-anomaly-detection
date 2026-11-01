"""Training package: losses, checkpoint management, and Trainer."""

from vad.training.checkpoints import load_checkpoint, save_checkpoint
from vad.training.losses import (
    AnomalyDetectionCriterion,
    LossBreakdown,
    temporal_smoothness_penalty,
    temporal_sparsity_penalty,
)
from vad.training.trainer import Trainer

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "AnomalyDetectionCriterion",
    "LossBreakdown",
    "temporal_smoothness_penalty",
    "temporal_sparsity_penalty",
    "Trainer",
]
