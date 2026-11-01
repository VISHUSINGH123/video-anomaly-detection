"""Timeline plotting and annotated video overlay generation for anomaly visualization."""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from vad.inference.video_scoring import VideoPredictionResult

# Use non-interactive Agg backend for headless safety
matplotlib.use("Agg")


def plot_anomaly_timeline(
    prediction: VideoPredictionResult,
    output_path: Optional[Path | str] = None,
    ground_truth_intervals: Optional[List[Tuple[int, int]]] = None,
) -> plt.Figure:
    """Create a publication-quality two-panel figure showing the anomaly timeline and prior gate weights.

    Args:
        prediction: VideoPredictionResult from the inference pipeline.
        output_path: Optional file path (`.png` or `.pdf`) to save the figure.
        ground_truth_intervals: Optional list of `(start_frame, end_frame)` ground-truth intervals.

    Returns:
        Matplotlib Figure instance.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3]}
    )

    times = prediction.frame_timestamps_sec
    scores = prediction.frame_scores
    thresh = prediction.threshold

    title_suffix = (
        " [SYNTHETIC SMOKE-TEST MODE]" if prediction.is_synthetic_mode else ""
    )
    ax1.set_title(
        f"Video Anomaly Score Timeline — {prediction.video_id}{title_suffix}",
        fontsize=12,
        fontweight="bold",
    )

    ax1.plot(times, scores, color="#1f77b4", linewidth=2.0, label="Predicted Anomaly Score")
    ax1.axhline(
        thresh,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Decision Threshold ({thresh:.2f})",
    )

    # Shade predicted high-score regions
    above = scores >= thresh
    ax1.fill_between(
        times,
        0,
        scores,
        where=above,
        color="#ff7f0e",
        alpha=0.35,
        label="Flagged Window (>= Threshold)",
    )

    # Shade ground-truth intervals if provided
    if ground_truth_intervals:
        for idx, (s_f, e_f) in enumerate(ground_truth_intervals):
            s_t = s_f / max(prediction.fps, 1e-3)
            e_t = e_f / max(prediction.fps, 1e-3)
            ax1.axvspan(
                s_t,
                e_t,
                color="#2ca02c",
                alpha=0.22,
                label="Ground-Truth Anomaly" if idx == 0 else None,
            )

    ax1.set_ylim(-0.02, 1.05)
    ax1.set_ylabel("Anomaly Score [0, 1]")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    # Bottom panel: Inspectable Multi-Prior Gate Weights over clips
    clip_times = [w.center_time_sec for w in prediction.windows]
    gates = prediction.gate_weights  # (num_clips, 4)
    labels = ["VideoMAE (Spatiotemporal)", "DINOv2 (Appearance)", "TCN (Temporal Context)", "Motion Velocity"]
    colors = ["#2b5c8f", "#388e3c", "#7b1fa2", "#e65100"]

    for k in range(min(4, gates.shape[1])):
        ax2.plot(
            clip_times,
            gates[:, k],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=colors[k],
            label=labels[k],
        )

    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Prior Gate Weight")
    ax2.set_title("Inspectable Multi-Prior Fusion Gate Weights", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")

    return fig


def export_annotated_video(
    frames_rgb: np.ndarray,
    prediction: VideoPredictionResult,
    output_video_path: Path | str,
) -> Path:
    """Export an MP4 video with a live anomaly-score bar and status banner overlay.

    Args:
        frames_rgb: Decoded RGB frames of shape `(N, H, W, 3)`, dtype uint8.
        prediction: VideoPredictionResult aligned with `frames_rgb`.
        output_video_path: Target `.mp4` path.

    Returns:
        Resolved Path to the written video file.
    """
    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_frames, h, w, _ = frames_rgb.shape
    # Ensure minimum canvas dimensions for readable text overlay
    target_w = max(w, 480)
    target_h = max(h, 360)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path), fourcc, float(max(prediction.fps, 1.0)), (target_w, target_h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open OpenCV VideoWriter for path: {out_path}")

    try:
        for i in range(num_frames):
            frame = frames_rgb[i]
            if frame.shape[0] != target_h or frame.shape[1] != target_w:
                frame = cv2.resize(
                    frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            score = float(prediction.frame_scores[min(i, len(prediction.frame_scores) - 1)])
            t_sec = float(
                prediction.frame_timestamps_sec[
                    min(i, len(prediction.frame_timestamps_sec) - 1)
                ]
            )
            is_flagged = score >= prediction.threshold

            # Top semi-transparent header bar
            overlay = bgr.copy()
            cv2.rectangle(overlay, (0, 0), (target_w, 58), (20, 20, 20), -1)
            bgr = cv2.addWeighted(overlay, 0.72, bgr, 0.28, 0)

            status_color = (0, 60, 255) if is_flagged else (60, 200, 60)
            status_label = "HIGH ANOMALY SCORE" if is_flagged else "NORMAL RANGE"
            if prediction.is_synthetic_mode:
                status_label += " (SYNTHETIC)"

            cv2.putText(
                bgr,
                f"Frame: {i:04d} | Time: {t_sec:05.2f}s | Score: {score:.3f}",
                (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                status_label,
                (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                2,
                cv2.LINE_AA,
            )

            # Progress / score meter on right side of header
            bar_x0, bar_y0 = target_w - 170, 16
            bar_w, bar_h = 150, 24
            cv2.rectangle(
                bgr,
                (bar_x0, bar_y0),
                (bar_x0 + bar_w, bar_y0 + bar_h),
                (180, 180, 180),
                1,
            )
            fill_w = int(np.clip(score, 0.0, 1.0) * (bar_w - 2))
            if fill_w > 0:
                cv2.rectangle(
                    bgr,
                    (bar_x0 + 1, bar_y0 + 1),
                    (bar_x0 + 1 + fill_w, bar_y0 + bar_h - 1),
                    status_color,
                    -1,
                )

            writer.write(bgr)
    finally:
        writer.release()

    return out_path
