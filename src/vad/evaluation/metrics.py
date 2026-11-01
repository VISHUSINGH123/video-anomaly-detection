"""Frame-level, clip-level, and video-level evaluation metrics for Video Anomaly Detection."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from vad.data.video_loader import TemporalWindow


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation summary across frames, clips, and videos."""

    num_videos: int
    num_frames: int
    num_clips: int
    threshold: float
    frame_roc_auc: Optional[float]
    frame_pr_auc: Optional[float]
    frame_precision: float
    frame_recall: float
    frame_f1: float
    frame_confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    clip_roc_auc: Optional[float]
    clip_pr_auc: Optional[float]
    video_roc_auc: Optional[float]
    is_synthetic: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def smooth_anomaly_scores(scores: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply edge-padded moving average smoothing to 1D temporal anomaly scores."""
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if window_size <= 1 or len(arr) <= 1:
        return arr.copy()
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return np.clip(smoothed.astype(np.float32), 0.0, 1.0)


def project_clip_scores_to_frames(
    clip_scores: Sequence[float] | np.ndarray,
    windows: Sequence[TemporalWindow],
    total_frames: int,
) -> np.ndarray:
    """Map overlapping clip-level anomaly scores to continuous frame-level anomaly scores.

    Frames covered by multiple overlapping windows receive the mean score across covering clips.
    Uncovered tail frames (if any) inherit the nearest valid frame score.

    Args:
        clip_scores: 1D sequence of clip anomaly scores in `[0, 1]`.
        windows: Sequence of TemporalWindow metadata aligned with `clip_scores`.
        total_frames: Total number of frames in the video.

    Returns:
        1D NumPy array of shape `(total_frames,)` with float32 frame-level scores.
    """
    if total_frames < 1:
        raise ValueError(f"total_frames must be >= 1, got {total_frames}")
    scores_arr = np.asarray(clip_scores, dtype=np.float32).reshape(-1)
    if len(scores_arr) != len(windows):
        raise ValueError(
            f"Length mismatch: {len(scores_arr)} clip scores vs {len(windows)} windows."
        )

    accum = np.zeros(total_frames, dtype=np.float64)
    counts = np.zeros(total_frames, dtype=np.float64)

    for score, win in zip(scores_arr, windows):
        for f_idx in win.frame_indices:
            if 0 <= f_idx < total_frames:
                accum[f_idx] += float(score)
                counts[f_idx] += 1.0

    frame_scores = np.zeros(total_frames, dtype=np.float32)
    covered = counts > 0
    frame_scores[covered] = (accum[covered] / counts[covered]).astype(np.float32)

    # Fill any uncovered gaps by nearest neighbor interpolation
    if not np.all(covered) and np.any(covered):
        valid_indices = np.where(covered)[0]
        missing_indices = np.where(~covered)[0]
        frame_scores[missing_indices] = np.interp(
            missing_indices, valid_indices, frame_scores[valid_indices]
        ).astype(np.float32)

    return np.clip(frame_scores, 0.0, 1.0)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute ROC-AUC safely, returning None when only a single class is present."""
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute Average Precision (PR-AUC) safely, returning None when no positive examples exist."""
    if np.sum(y_true > 0) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def compute_anomaly_metrics(
    frame_labels_all: np.ndarray,
    frame_scores_all: np.ndarray,
    clip_labels_all: Optional[np.ndarray] = None,
    clip_scores_all: Optional[np.ndarray] = None,
    video_labels_all: Optional[np.ndarray] = None,
    video_scores_all: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    num_videos: int = 1,
    is_synthetic: bool = False,
) -> EvaluationMetrics:
    """Compute frame-level, clip-level, and video-level evaluation metrics with edge-case safety."""
    y_true_f = (np.asarray(frame_labels_all).reshape(-1) > 0.5).astype(np.int64)
    y_score_f = np.asarray(frame_scores_all, dtype=np.float64).reshape(-1)

    if len(y_true_f) != len(y_score_f) or len(y_true_f) == 0:
        raise ValueError(
            f"Invalid frame label/score lengths: {len(y_true_f)} vs {len(y_score_f)}"
        )

    y_pred_f = (y_score_f >= threshold).astype(np.int64)

    frame_roc = _safe_roc_auc(y_true_f, y_score_f)
    frame_pr = _safe_pr_auc(y_true_f, y_score_f)

    prec = float(precision_score(y_true_f, y_pred_f, zero_division=0))
    rec = float(recall_score(y_true_f, y_pred_f, zero_division=0))
    f1 = float(f1_score(y_true_f, y_pred_f, zero_division=0))
    cm = confusion_matrix(y_true_f, y_pred_f, labels=[0, 1]).tolist()

    clip_roc: Optional[float] = None
    clip_pr: Optional[float] = None
    num_clips = 0
    if clip_labels_all is not None and clip_scores_all is not None and len(clip_labels_all) > 0:
        y_true_c = (np.asarray(clip_labels_all).reshape(-1) > 0.5).astype(np.int64)
        y_score_c = np.asarray(clip_scores_all, dtype=np.float64).reshape(-1)
        num_clips = len(y_true_c)
        clip_roc = _safe_roc_auc(y_true_c, y_score_c)
        clip_pr = _safe_pr_auc(y_true_c, y_score_c)

    video_roc: Optional[float] = None
    if video_labels_all is not None and video_scores_all is not None and len(video_labels_all) > 0:
        y_true_v = (np.asarray(video_labels_all).reshape(-1) > 0.5).astype(np.int64)
        y_score_v = np.asarray(video_scores_all, dtype=np.float64).reshape(-1)
        video_roc = _safe_roc_auc(y_true_v, y_score_v)

    notes_list = []
    if frame_roc is None:
        notes_list.append("Frame ROC-AUC undefined (ground truth contains only a single class).")
    if is_synthetic:
        notes_list.append(
            "SYNTHETIC DATA EVALUATION: Metrics computed on synthetic smoke-test data, not real surveillance benchmarks."
        )

    return EvaluationMetrics(
        num_videos=int(num_videos),
        num_frames=int(len(y_true_f)),
        num_clips=int(num_clips),
        threshold=float(threshold),
        frame_roc_auc=frame_roc,
        frame_pr_auc=frame_pr,
        frame_precision=prec,
        frame_recall=rec,
        frame_f1=f1,
        frame_confusion_matrix=[[int(x) for x in row] for row in cm],
        clip_roc_auc=clip_roc,
        clip_pr_auc=clip_pr,
        video_roc_auc=video_roc,
        is_synthetic=bool(is_synthetic),
        notes=" ".join(notes_list),
    )
