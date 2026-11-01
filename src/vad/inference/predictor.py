"""Streaming-compatible sliding-window video anomaly predictor."""

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from vad.config import VADConfig
from vad.data.video_loader import DecodedVideo, build_temporal_windows, decode_video
from vad.device import resolve_device
from vad.evaluation.metrics import project_clip_scores_to_frames, smooth_anomaly_scores
from vad.inference.video_scoring import VideoPredictionResult
from vad.models.anomaly_detector import AnomalyDetector
from vad.models.dinov2_encoder import DINOv2Encoder
from vad.models.videomae_encoder import VideoMAEEncoder
from vad.training.checkpoints import load_checkpoint
from vad.utils.logging import setup_logger

logger = setup_logger("vad.predictor")


def _extract_deterministic_synthetic_features(
    clips_rgb: np.ndarray, feature_dim: int = 768
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract deterministic image-statistics features ONLY when `--synthetic-encoders` is explicitly requested.

    This is used exclusively for offline smoke testing without downloading ~700 MB of Hugging Face weights,
    and sets `is_synthetic_mode = True` on all outputs.
    """
    num_clips = clips_rgb.shape[0]
    v_feats = np.zeros((num_clips, feature_dim), dtype=np.float32)
    d_feats = np.zeros((num_clips, feature_dim), dtype=np.float32)

    for i in range(num_clips):
        clip = clips_rgb[i].astype(np.float32) / 255.0  # (T, H, W, 3)
        # Temporal frame-to-frame motion statistics for spatiotemporal stream
        diffs = np.abs(np.diff(clip, axis=0))
        motion_energy = float(np.mean(diffs))
        motion_max = float(np.max(diffs))
        spatial_mean = np.mean(clip, axis=(0, 1, 2))  # (3,)
        spatial_std = np.std(clip, axis=(0, 1, 2))  # (3,)

        # Construct deterministic feature vector from frame statistics
        base_v = np.array(
            [
                motion_energy * 10.0,
                motion_max * 5.0,
                spatial_mean[0],
                spatial_mean[1],
                spatial_mean[2],
                spatial_std[0] * 4.0,
                spatial_std[1] * 4.0,
                spatial_std[2] * 4.0,
            ],
            dtype=np.float32,
        )
        v_feats[i] = np.resize(base_v, (feature_dim,))

        # Frame-level appearance histogram statistics for DINOv2 stream
        hist_r, _ = np.histogram(clip[:, :, :, 0], bins=16, range=(0.0, 1.0), density=True)
        hist_g, _ = np.histogram(clip[:, :, :, 1], bins=16, range=(0.0, 1.0), density=True)
        hist_b, _ = np.histogram(clip[:, :, :, 2], bins=16, range=(0.0, 1.0), density=True)
        base_d = np.concatenate([hist_r, hist_g, hist_b, spatial_mean, spatial_std]).astype(
            np.float32
        )
        d_feats[i] = np.resize(base_d, (feature_dim,))

    return v_feats, d_feats


class VideoAnomalyPredictor:
    """Near-real-time / sliding-window video anomaly inference engine."""

    def __init__(
        self,
        model: AnomalyDetector,
        config: VADConfig,
        device: Optional[torch.device] = None,
        videomae_encoder: Optional[VideoMAEEncoder] = None,
        dinov2_encoder: Optional[DINOv2Encoder] = None,
        use_synthetic_encoders: bool = False,
        checkpoint_is_synthetic: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or resolve_device(config.device).device
        self.model.to(self.device)
        self.model.eval()

        self.use_synthetic_encoders = use_synthetic_encoders
        self.checkpoint_is_synthetic = checkpoint_is_synthetic

        self.videomae_encoder = videomae_encoder
        self.dinov2_encoder = dinov2_encoder

        if not self.use_synthetic_encoders:
            if self.config.encoders.videomae.enabled and self.videomae_encoder is None:
                self.videomae_encoder = VideoMAEEncoder(
                    config=self.config.encoders.videomae, device=self.device
                )
            if self.config.encoders.dinov2.enabled and self.dinov2_encoder is None:
                self.dinov2_encoder = DINOv2Encoder(
                    config=self.config.encoders.dinov2, device=self.device
                )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        device_override: Optional[str] = None,
        use_synthetic_encoders: bool = False,
    ) -> "VideoAnomalyPredictor":
        """Instantiate a VideoAnomalyPredictor directly from a saved `.pt` checkpoint."""
        dev_info = resolve_device(device_override)
        model, config, meta = load_checkpoint(checkpoint_path, device=dev_info.device)
        if device_override:
            config.device = device_override
        return cls(
            model=model,
            config=config,
            device=dev_info.device,
            use_synthetic_encoders=use_synthetic_encoders,
            checkpoint_is_synthetic=bool(meta.get("is_synthetic", False)),
        )

    @torch.inference_mode()
    def predict_decoded_video(
        self,
        decoded: DecodedVideo,
        window_length: Optional[int] = None,
        window_stride: Optional[int] = None,
        threshold: Optional[float] = None,
        smoothing_window: Optional[int] = None,
    ) -> Tuple[VideoPredictionResult, DecodedVideo]:
        """Run sliding-window feature extraction, temporal modeling, and fusion on a DecodedVideo."""
        t0 = time.perf_counter()

        win_len = window_length or self.config.inference.window_length
        win_stride = window_stride or self.config.inference.window_stride
        thresh = threshold if threshold is not None else self.config.inference.threshold
        smooth_w = (
            smoothing_window
            if smoothing_window is not None
            else self.config.inference.smoothing_window
        )

        clips_rgb, windows = build_temporal_windows(
            decoded,
            clip_length=win_len,
            clip_stride=win_stride,
            short_video_policy="pad",
        )
        num_clips = len(windows)

        if self.use_synthetic_encoders:
            v_np, d_np = _extract_deterministic_synthetic_features(
                clips_rgb,
                feature_dim=self.config.encoders.videomae.feature_dim,
            )
        else:
            if self.config.encoders.videomae.enabled and self.videomae_encoder is not None:
                v_np = self.videomae_encoder.extract_clips(clips_rgb)
            else:
                v_np = np.zeros(
                    (num_clips, self.config.encoders.videomae.feature_dim),
                    dtype=np.float32,
                )

            if self.config.encoders.dinov2.enabled and self.dinov2_encoder is not None:
                d_np, _ = self.dinov2_encoder.extract_clips(clips_rgb)
            else:
                d_np = np.zeros(
                    (num_clips, self.config.encoders.dinov2.feature_dim),
                    dtype=np.float32,
                )

        v_tensor = torch.from_numpy(v_np).unsqueeze(0).to(self.device)  # (1, T, D_v)
        d_tensor = torch.from_numpy(d_np).unsqueeze(0).to(self.device)  # (1, T, D_d)
        mask = torch.ones((1, num_clips), dtype=torch.bool, device=self.device)

        fusion_out = self.model(v_tensor, d_tensor, mask=mask)
        raw_scores = fusion_out.scores[0].detach().cpu().numpy()
        smoothed_scores = smooth_anomaly_scores(raw_scores, window_size=smooth_w)
        gates = fusion_out.gate_weights[0].detach().cpu().numpy()
        motion_mag = fusion_out.motion_magnitude[0].detach().cpu().numpy()

        total_frames = decoded.frames_rgb.shape[0]
        frame_scores = project_clip_scores_to_frames(
            smoothed_scores, windows, total_frames=total_frames
        )

        elapsed = max(1e-6, time.perf_counter() - t0)
        eff_fps = float(total_frames / elapsed)
        duration_sec = float(decoded.timestamps_sec[-1]) if decoded.timestamps_sec else 0.0

        result = VideoPredictionResult(
            video_id=decoded.video_id,
            video_path=str(decoded.source_path),
            fps=decoded.fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            windows=windows,
            raw_clip_scores=raw_scores,
            smoothed_clip_scores=smoothed_scores,
            frame_scores=frame_scores,
            frame_timestamps_sec=np.asarray(decoded.timestamps_sec, dtype=np.float32),
            gate_weights=gates,
            prior_names=fusion_out.prior_names,
            motion_magnitude=motion_mag,
            threshold=float(thresh),
            inference_time_sec=round(elapsed, 4),
            effective_fps=round(eff_fps, 2),
            is_synthetic_mode=bool(self.use_synthetic_encoders or self.checkpoint_is_synthetic),
        )
        return result, decoded

    def predict_video_file(
        self,
        video_path: Path | str,
        window_length: Optional[int] = None,
        window_stride: Optional[int] = None,
        threshold: Optional[float] = None,
        smoothing_window: Optional[int] = None,
        max_frames: Optional[int] = None,
        max_dimension: Optional[int] = None,
    ) -> Tuple[VideoPredictionResult, DecodedVideo]:
        """Decode a local video file and run full sliding-window anomaly prediction.

        Args:
            video_path: Path to video file or directory.
            window_length: Number of frames per temporal clip.
            window_stride: Frame step between consecutive clips.
            threshold: Anomaly score decision threshold.
            smoothing_window: Temporal moving-average smoothing kernel size.
            max_frames: Optional cap on decoded frames.
            max_dimension: Optional maximum frame dimension (width or height) to limit RAM usage.
        """
        resolved_max_dim = (
            max_dimension
            if max_dimension is not None
            else getattr(self.config.data, "max_dimension", 384)
        )
        decoded = decode_video(
            video_path,
            target_fps=self.config.data.target_fps,
            max_frames=max_frames,
            max_dimension=resolved_max_dim,
        )
        return self.predict_decoded_video(
            decoded=decoded,
            window_length=window_length,
            window_stride=window_stride,
            threshold=threshold,
            smoothing_window=smoothing_window,
        )
