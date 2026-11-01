"""Evaluation package: frame/clip/video metrics and dataset Evaluator."""

from vad.evaluation.evaluator import Evaluator
from vad.evaluation.metrics import (
    EvaluationMetrics,
    compute_anomaly_metrics,
    project_clip_scores_to_frames,
    smooth_anomaly_scores,
)

__all__ = [
    "Evaluator",
    "EvaluationMetrics",
    "compute_anomaly_metrics",
    "project_clip_scores_to_frames",
    "smooth_anomaly_scores",
]
