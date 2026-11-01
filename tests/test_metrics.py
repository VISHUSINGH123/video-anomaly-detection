"""Unit tests for evaluation metrics, frame-score projection, and single-class edge cases."""

import numpy as np

from vad.data.video_loader import TemporalWindow
from vad.evaluation.metrics import (
    compute_anomaly_metrics,
    project_clip_scores_to_frames,
    smooth_anomaly_scores,
)


def test_compute_anomaly_metrics_known_toy_example() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    y_score = np.array([0.05, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.95], dtype=np.float32)

    metrics = compute_anomaly_metrics(
        frame_labels_all=y_true,
        frame_scores_all=y_score,
        clip_labels_all=y_true,
        clip_scores_all=y_score,
        video_labels_all=np.array([0, 1]),
        video_scores_all=np.array([0.2, 0.9]),
        threshold=0.5,
        num_videos=2,
    )

    assert metrics.frame_roc_auc == 1.0
    assert metrics.frame_pr_auc == 1.0
    assert metrics.frame_precision == 1.0
    assert metrics.frame_recall == 1.0
    assert metrics.frame_f1 == 1.0
    assert metrics.frame_confusion_matrix == [[4, 0], [0, 4]]
    assert metrics.video_roc_auc == 1.0


def test_single_class_edge_case_handling() -> None:
    """Verify that videos containing only normal frames do not crash ROC-AUC / PR-AUC calculation."""
    y_true_only_normal = np.zeros(20, dtype=np.int64)
    y_score = np.linspace(0.01, 0.4, num=20, dtype=np.float32)

    metrics = compute_anomaly_metrics(
        frame_labels_all=y_true_only_normal,
        frame_scores_all=y_score,
        threshold=0.5,
    )
    assert metrics.frame_roc_auc is None
    assert metrics.frame_pr_auc is None
    assert metrics.frame_confusion_matrix == [[20, 0], [0, 0]]
    assert "single class" in metrics.notes.lower()


def test_project_clip_scores_to_frames_and_smoothing() -> None:
    w0 = TemporalWindow(
        clip_index=0,
        frame_indices=[0, 1, 2, 3],
        start_frame=0,
        end_frame=3,
        start_time_sec=0.0,
        end_time_sec=0.3,
        center_time_sec=0.15,
        is_padded=False,
    )
    w1 = TemporalWindow(
        clip_index=1,
        frame_indices=[2, 3, 4, 5],
        start_frame=2,
        end_frame=5,
        start_time_sec=0.2,
        end_time_sec=0.5,
        center_time_sec=0.35,
        is_padded=False,
    )

    frame_scores = project_clip_scores_to_frames([0.2, 0.8], [w0, w1], total_frames=6)
    # Frames 0, 1 covered only by w0 -> 0.2
    # Frames 2, 3 covered by both w0 and w1 -> (0.2 + 0.8) / 2 = 0.5
    # Frames 4, 5 covered only by w1 -> 0.8
    expected = np.array([0.2, 0.2, 0.5, 0.5, 0.8, 0.8], dtype=np.float32)
    assert np.allclose(frame_scores, expected, atol=1e-5)

    smoothed = smooth_anomaly_scores(np.array([0.0, 1.0, 0.0], dtype=np.float32), window_size=3)
    assert smoothed.shape == (3,)
    assert smoothed[1] < 1.0 and smoothed[0] > 0.0
