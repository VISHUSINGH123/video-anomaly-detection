"""Dataset-level evaluator producing per-video timelines and JSON/CSV evaluation summaries."""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from vad.config import VADConfig
from vad.data.annotations import VideoAnnotation
from vad.data.dataset import CachedFeatureDataset, collate_variable_length_features
from vad.evaluation.metrics import (
    EvaluationMetrics,
    compute_anomaly_metrics,
    project_clip_scores_to_frames,
    smooth_anomaly_scores,
)
from vad.models.anomaly_detector import AnomalyDetector
from vad.utils.logging import setup_logger

logger = setup_logger("vad.evaluator")


class Evaluator:
    """Evaluates a trained AnomalyDetector on a dataset of cached video features and annotations."""

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
        self.model.eval()

    @torch.inference_mode()
    def evaluate_dataset(
        self,
        dataset: CachedFeatureDataset,
        annotations: Optional[Dict[str, VideoAnnotation]] = None,
        output_dir: Optional[Path | str] = None,
    ) -> Tuple[EvaluationMetrics, Dict[str, Any]]:
        """Run evaluation over all videos in `dataset` and optionally export JSON/CSV reports."""
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_variable_length_features,
        )

        all_frame_labels: List[np.ndarray] = []
        all_frame_scores: List[np.ndarray] = []
        all_clip_labels: List[np.ndarray] = []
        all_clip_scores: List[np.ndarray] = []
        all_video_labels: List[int] = []
        all_video_scores: List[float] = []
        per_video_records: List[Dict[str, Any]] = []
        any_synthetic = False

        for batch in loader:
            vid_id = batch["video_ids"][0]
            v_feat = batch["videomae_features"].to(self.device)
            d_feat = batch["dinov2_features"].to(self.device)
            mask = batch["mask"].to(self.device)
            windows = batch["windows"][0]
            total_frames = max(1, int(batch["original_num_frames"][0]))
            any_synthetic = any_synthetic or bool(batch["is_synthetic"])

            out = self.model(v_feat, d_feat, mask=mask)
            raw_clip_scores = out.scores[0, : len(windows)].detach().cpu().numpy()
            smoothed_clip_scores = smooth_anomaly_scores(
                raw_clip_scores, window_size=self.config.evaluation.smoothing_window
            )
            gate_weights = out.gate_weights[0, : len(windows)].detach().cpu().numpy()
            mean_gates = gate_weights.mean(axis=0).tolist() if len(windows) > 0 else [0.25] * 4

            frame_scores = project_clip_scores_to_frames(
                smoothed_clip_scores, windows, total_frames=total_frames
            )

            clip_gt = batch["clip_labels"][0, : len(windows)].numpy()
            video_gt = int(batch["video_labels"][0].item())

            if annotations and vid_id in annotations:
                ann = annotations[vid_id]
                frame_gt = ann.get_frame_labels(total_frames)
                video_gt = int(ann.video_label)
            else:
                # Reconstruct frame-level ground truth from clip-level ground truth when explicit
                # frame interval dictionary is not separately provided
                frame_gt = (
                    project_clip_scores_to_frames(clip_gt, windows, total_frames=total_frames)
                    >= 0.5
                ).astype(np.int64)

            # Video-level score is Top-K mean clip score
            k = max(
                1,
                int(math.ceil(self.config.training.mil_topk_ratio * len(smoothed_clip_scores))),
            )
            video_score = float(np.mean(np.sort(smoothed_clip_scores)[-k:]))

            all_frame_labels.append(frame_gt)
            all_frame_scores.append(frame_scores)
            all_clip_labels.append(clip_gt)
            all_clip_scores.append(smoothed_clip_scores)
            all_video_labels.append(video_gt)
            all_video_scores.append(video_score)

            per_video_records.append(
                {
                    "video_id": vid_id,
                    "video_label": video_gt,
                    "video_score": round(video_score, 6),
                    "max_clip_score": round(float(np.max(smoothed_clip_scores)), 6),
                    "mean_clip_score": round(float(np.mean(smoothed_clip_scores)), 6),
                    "num_clips": len(windows),
                    "num_frames": total_frames,
                    "mean_gate_videomae": round(float(mean_gates[0]), 4),
                    "mean_gate_dinov2": round(float(mean_gates[1]), 4),
                    "mean_gate_tcn": round(float(mean_gates[2]), 4),
                    "mean_gate_motion": round(float(mean_gates[3]), 4),
                    "clip_scores": [round(float(s), 6) for s in smoothed_clip_scores],
                }
            )

        metrics = compute_anomaly_metrics(
            frame_labels_all=np.concatenate(all_frame_labels, axis=0),
            frame_scores_all=np.concatenate(all_frame_scores, axis=0),
            clip_labels_all=np.concatenate(all_clip_labels, axis=0),
            clip_scores_all=np.concatenate(all_clip_scores, axis=0),
            video_labels_all=np.asarray(all_video_labels, dtype=np.int64),
            video_scores_all=np.asarray(all_video_scores, dtype=np.float64),
            threshold=self.config.evaluation.threshold,
            num_videos=len(per_video_records),
            is_synthetic=any_synthetic,
        )

        details = {
            "experiment_name": self.config.experiment_name,
            "fusion_mode": self.config.fusion.mode,
            "is_synthetic": any_synthetic,
            "metrics": metrics.to_dict(),
            "per_video": per_video_records,
        }

        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            json_file = out_path / "evaluation_summary.json"
            csv_file = out_path / "per_video_summary.csv"

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(details, f, indent=2)

            if per_video_records:
                fieldnames = [
                    k for k in per_video_records[0].keys() if k != "clip_scores"
                ]
                with open(csv_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in per_video_records:
                        writer.writerow({k: row[k] for k in fieldnames})

            logger.info("Saved evaluation reports to %s and %s", json_file, csv_file)

        return metrics, details
