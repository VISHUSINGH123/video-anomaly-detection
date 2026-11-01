"""Temporal window scoring, frame-level score projection, and CSV/DataFrame export."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from vad.data.video_loader import TemporalWindow
from vad.evaluation.metrics import project_clip_scores_to_frames, smooth_anomaly_scores


@dataclass
class VideoPredictionResult:
    """Complete result of running anomaly inference on a single video."""

    video_id: str
    video_path: str
    fps: float
    total_frames: int
    duration_sec: float
    windows: List[TemporalWindow]
    raw_clip_scores: np.ndarray  # (num_clips,)
    smoothed_clip_scores: np.ndarray  # (num_clips,)
    frame_scores: np.ndarray  # (total_frames,)
    frame_timestamps_sec: np.ndarray  # (total_frames,)
    gate_weights: np.ndarray  # (num_clips, 4)
    prior_names: tuple[str, ...]
    motion_magnitude: np.ndarray  # (num_clips,)
    threshold: float
    inference_time_sec: float
    effective_fps: float
    is_synthetic_mode: bool = False

    def to_clip_dataframe(self) -> pd.DataFrame:
        """Build a timestamped DataFrame of clip-level anomaly scores and prior gate weights."""
        rows: List[Dict[str, Any]] = []
        for idx, win in enumerate(self.windows):
            raw_s = float(self.raw_clip_scores[idx])
            smooth_s = float(self.smoothed_clip_scores[idx])
            gates = self.gate_weights[idx]
            rows.append(
                {
                    "clip_index": win.clip_index,
                    "start_frame": win.start_frame,
                    "end_frame": win.end_frame,
                    "start_time_sec": round(win.start_time_sec, 3),
                    "end_time_sec": round(win.end_time_sec, 3),
                    "center_time_sec": round(win.center_time_sec, 3),
                    "raw_anomaly_score": round(raw_s, 6),
                    "anomaly_score": round(smooth_s, 6),
                    "is_flagged": bool(smooth_s >= self.threshold),
                    "gate_videomae": round(float(gates[0]), 4),
                    "gate_dinov2": round(float(gates[1]), 4),
                    "gate_tcn_temporal": round(float(gates[2]), 4),
                    "gate_motion_dynamics": round(float(gates[3]), 4),
                    "motion_magnitude": round(float(self.motion_magnitude[idx]), 4),
                    "is_synthetic_mode": self.is_synthetic_mode,
                }
            )
        return pd.DataFrame(rows)

    def to_frame_dataframe(self) -> pd.DataFrame:
        """Build a frame-level DataFrame mapping every frame index and timestamp to its anomaly score."""
        return pd.DataFrame(
            {
                "frame_index": np.arange(self.total_frames, dtype=np.int64),
                "timestamp_sec": np.round(self.frame_timestamps_sec, 4),
                "anomaly_score": np.round(self.frame_scores, 6),
                "is_flagged": self.frame_scores >= self.threshold,
                "is_synthetic_mode": self.is_synthetic_mode,
            }
        )

    def save_csv(self, output_dir: Path | str) -> tuple[Path, Path]:
        """Export both clip-level and frame-level timestamped CSV files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_csv = out_dir / f"{self.video_id}_clip_scores.csv"
        frame_csv = out_dir / f"{self.video_id}_frame_scores.csv"
        self.to_clip_dataframe().to_csv(clip_csv, index=False)
        self.to_frame_dataframe().to_csv(frame_csv, index=False)
        return clip_csv, frame_csv
