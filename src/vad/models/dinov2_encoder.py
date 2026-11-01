"""Pretrained DINOv2 frame-level visual encoder (`facebook/dinov2-base`).

Model documentation:
- Checkpoint: `facebook/dinov2-base`
- License: Apache-2.0 (Meta Platforms, Inc.)
- Input tensor shape: `(N_frames, C=3, H=224, W=224)` normalized with ImageNet mean/std
- Output representation:
  - Per-frame CLS token embedding (`last_hidden_state[:, 0, :]`) of dimension `D = 768`
  - Aggregated across sampled frames within each temporal clip via temporal mean (`cls`)
    or mean + temporal standard deviation modulation (`cls_and_std`), producing a clip-aligned
    representation of shape `(num_clips, 768)`.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from vad.config import DINOv2Config
from vad.data.preprocessing import sample_clip_frames_for_dinov2
from vad.utils.logging import setup_logger

logger = setup_logger("vad.dinov2")


class DINOv2Encoder(nn.Module):
    """Frame-level appearance feature extractor with clip-level temporal aggregation."""

    def __init__(
        self,
        config: Optional[DINOv2Config] = None,
        device: Optional[torch.device] = None,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the DINOv2 encoder.

        Args:
            config: DINOv2Config parameters.
            device: Target compute device.
            backbone: Optional pre-instantiated `Dinov2Model` (used in unit tests without network calls).
        """
        super().__init__()
        self.config = config or DINOv2Config()
        self.device = device or torch.device("cpu")
        self.backbone: Optional[nn.Module] = backbone

        if self.backbone is not None:
            self.backbone.to(self.device)
            self.backbone.eval()

    def load_pretrained(self) -> None:
        """Load the pretrained DINOv2 checkpoint from Hugging Face or local cache."""
        if self.backbone is not None:
            return
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required to load DINOv2. "
                "Install it via `pip install transformers`."
            ) from exc

        logger.info(
            "Loading pretrained DINOv2 checkpoint '%s' onto %s...",
            self.config.model_name,
            self.device,
        )
        try:
            # Try loading directly from local Hugging Face cache first for instant offline loading
            model = AutoModel.from_pretrained(
                self.config.model_name,
                local_files_only=True,
            )
        except Exception:
            # Fallback to online download if not already cached
            logger.info("Not in local cache. Downloading '%s' from Hugging Face Hub...", self.config.model_name)
            try:
                model = AutoModel.from_pretrained(
                    self.config.model_name,
                    local_files_only=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load DINOv2 checkpoint '{self.config.model_name}'. "
                    "Ensure internet connectivity for initial download (~346 MB) or cache the checkpoint locally. "
                    f"Original error: {exc}"
                ) from exc

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.to(self.device)
        self.backbone = model

    @torch.inference_mode()
    def forward_frames(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract CLS embeddings for a batch of 2D frames.

        Args:
            pixel_values: Float tensor of shape `(N, 3, H, W)`.

        Returns:
            Float32 tensor of shape `(N, feature_dim)` on CPU.
        """
        if self.backbone is None:
            self.load_pretrained()
        assert self.backbone is not None

        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError(
                f"DINOv2 expects pixel_values of shape (N, 3, H, W), got {tuple(pixel_values.shape)}"
            )

        inputs = pixel_values.to(self.device, non_blocking=True)
        use_amp = self.config.use_fp16 and self.device.type == "cuda"

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.backbone(pixel_values=inputs)
                cls_tokens = outputs.last_hidden_state[:, 0, :]
        else:
            outputs = self.backbone(pixel_values=inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :]

        feats_cpu = cls_tokens.detach().float().cpu()
        if not torch.isfinite(feats_cpu).all():
            raise RuntimeError("Non-finite values (NaN/Inf) detected in DINOv2 output features.")
        return feats_cpu

    def aggregate_clip_frames(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregate `K` sampled frame embeddings `(K, D)` into a single clip representation `(D,)`.

        If `pooling == "cls"`, computes the temporal mean across sampled frames.
        If `pooling == "cls_and_std"`, combines the temporal mean and intra-clip temporal standard
        deviation (`mean + 0.5 * std`) to preserve both static appearance and rapid appearance change.
        """
        if frame_embeddings.ndim != 2:
            raise ValueError(
                f"Expected frame_embeddings of shape (K, D), got {tuple(frame_embeddings.shape)}"
            )
        mean_feat = frame_embeddings.mean(dim=0)
        if self.config.pooling == "cls" or frame_embeddings.shape[0] < 2:
            return mean_feat
        elif self.config.pooling == "cls_and_std":
            std_feat = frame_embeddings.std(dim=0, unbiased=False)
            return mean_feat + 0.5 * std_feat
        else:
            raise ValueError(f"Unsupported DINOv2 pooling mode: {self.config.pooling}")

    @torch.inference_mode()
    def extract_clips(
        self, clips_rgb: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """Extract clip-aligned DINOv2 features from an array or sequence of RGB video clips.

        Args:
            clips_rgb: NumPy array or ClipWindowArray of shape `(num_clips, clip_len, H, W, 3)`, dtype uint8.

        Returns:
            Tuple of:
              - clip_features: NumPy array of shape `(num_clips, feature_dim)`, dtype float32
              - sampled_relative_indices: List of relative frame indices sampled per clip
        """
        if clips_rgb.ndim != 5 or clips_rgb.shape[-1] != 3:
            raise ValueError(
                f"Expected clips_rgb of shape (num_clips, T, H, W, 3), got {clips_rgb.shape}"
            )
        num_clips = clips_rgb.shape[0]
        if num_clips == 0:
            return np.empty((0, self.config.feature_dim), dtype=np.float32), []

        sampled_tensors: List[torch.Tensor] = []
        sampled_indices_per_clip: List[List[int]] = []
        counts_per_clip: List[int] = []

        for i in range(num_clips):
            f_tensor, rel_idx = sample_clip_frames_for_dinov2(
                clips_rgb[i],
                frame_sample_rate=self.config.frame_sample_rate,
                image_size=self.config.image_size,
            )
            sampled_tensors.append(f_tensor)
            sampled_indices_per_clip.append(rel_idx)
            counts_per_clip.append(f_tensor.shape[0])

        all_frames = torch.cat(sampled_tensors, dim=0)  # (Total_K, 3, H, W)
        batch_size = max(1, self.config.batch_size)
        extracted_chunks: List[torch.Tensor] = []

        for start_idx in range(0, all_frames.shape[0], batch_size):
            end_idx = min(all_frames.shape[0], start_idx + batch_size)
            chunk = all_frames[start_idx:end_idx]
            extracted_chunks.append(self.forward_frames(chunk))

        all_frame_feats = torch.cat(extracted_chunks, dim=0)  # (Total_K, D)

        clip_features_list: List[torch.Tensor] = []
        cursor = 0
        for count in counts_per_clip:
            sub_feats = all_frame_feats[cursor : cursor + count]
            cursor += count
            clip_features_list.append(self.aggregate_clip_frames(sub_feats))

        result = torch.stack(clip_features_list, dim=0).numpy().astype(np.float32)
        if result.shape != (num_clips, self.config.feature_dim):
            raise RuntimeError(
                f"Unexpected DINOv2 output shape {result.shape}, expected {(num_clips, self.config.feature_dim)}"
            )
        return result, sampled_indices_per_clip
