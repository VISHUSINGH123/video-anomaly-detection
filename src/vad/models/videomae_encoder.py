"""Pretrained VideoMAE spatiotemporal clip encoder (`MCG-NJU/videomae-base`).

Model documentation:
- Checkpoint: `MCG-NJU/videomae-base` (or `MCG-NJU/videomae-base-finetuned-kinetics`)
- License: CC-BY-NC 4.0 (official MCG-NJU release) / Apache-2.0 Transformers code
- Input tensor shape: `(B, T=16, C=3, H=224, W=224)`
- Spatiotemporal tubelet patching: tubelet_size=2, patch_size=16x16 -> `(16/2) * (224/16) * (224/16) = 8 * 14 * 14 = 1568` tokens
- Output representation: Mean-pooled or max-pooled token embedding over the 1568 spatiotemporal tokens -> `(B, 768)`
- Temporal receptive field: 16 sampled video frames per clip
"""

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from vad.config import VideoMAEConfig
from vad.data.preprocessing import preprocess_clip_for_videomae
from vad.utils.logging import setup_logger

logger = setup_logger("vad.videomae")


class VideoMAEEncoder(nn.Module):
    """Reusable spatiotemporal feature extractor built on Hugging Face VideoMAEModel."""

    def __init__(
        self,
        config: Optional[VideoMAEConfig] = None,
        device: Optional[torch.device] = None,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the VideoMAE encoder.

        Args:
            config: VideoMAEConfig parameters.
            device: Target execution device (CPU, CUDA, DirectML, MPS).
            backbone: Optional pre-instantiated `VideoMAEModel` (used for testing without network downloads).
        """
        super().__init__()
        self.config = config or VideoMAEConfig()
        self.device = device or torch.device("cpu")
        self.backbone: Optional[nn.Module] = backbone

        if self.backbone is not None:
            self.backbone.to(self.device)
            self.backbone.eval()

    def load_pretrained(self) -> None:
        """Load the pretrained VideoMAEModel checkpoint from Hugging Face or local cache."""
        if self.backbone is not None:
            return
        try:
            from transformers import VideoMAEModel
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required to load VideoMAE. "
                "Install it via `pip install transformers`."
            ) from exc

        logger.info(
            "Loading pretrained VideoMAE checkpoint '%s' onto %s...",
            self.config.model_name,
            self.device,
        )
        try:
            # Try loading directly from local Hugging Face cache first for instant offline loading
            model = VideoMAEModel.from_pretrained(
                self.config.model_name,
                local_files_only=True,
            )
        except Exception:
            # Fallback to online download if not already cached
            logger.info("Not in local cache. Downloading '%s' from Hugging Face Hub...", self.config.model_name)
            try:
                model = VideoMAEModel.from_pretrained(
                    self.config.model_name,
                    local_files_only=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load VideoMAE checkpoint '{self.config.model_name}'. "
                    "Ensure internet connectivity for initial download (~345 MB) or cache the checkpoint locally. "
                    f"Original error: {exc}"
                ) from exc

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.to(self.device)
        self.backbone = model

    def _pool_tokens(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """Pool spatiotemporal patch tokens `(B, N_tokens, D)` into clip features `(B, D)`."""
        if last_hidden_state.ndim != 3:
            raise ValueError(
                f"Expected VideoMAE last_hidden_state of shape (B, N, D), got {tuple(last_hidden_state.shape)}"
            )
        if self.config.pooling == "mean":
            pooled = last_hidden_state.mean(dim=1)
        elif self.config.pooling == "max":
            pooled = last_hidden_state.max(dim=1).values
        else:
            raise ValueError(f"Unsupported VideoMAE pooling mode: {self.config.pooling}")

        return pooled

    @torch.inference_mode()
    def forward_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass on a batch of preprocessed clips.

        Args:
            pixel_values: Float tensor of shape `(B, T, 3, H, W)`.

        Returns:
            Float32 tensor of shape `(B, feature_dim)` on CPU.
        """
        if self.backbone is None:
            self.load_pretrained()
        assert self.backbone is not None

        if pixel_values.ndim != 5 or pixel_values.shape[2] != 3:
            raise ValueError(
                f"VideoMAE expects pixel_values of shape (B, T, 3, H, W), got {tuple(pixel_values.shape)}"
            )

        inputs = pixel_values.to(self.device, non_blocking=True)
        use_amp = self.config.use_fp16 and self.device.type == "cuda"

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.backbone(pixel_values=inputs)
                feats = self._pool_tokens(outputs.last_hidden_state)
        else:
            outputs = self.backbone(pixel_values=inputs)
            feats = self._pool_tokens(outputs.last_hidden_state)

        feats_cpu = feats.detach().float().cpu()
        if not torch.isfinite(feats_cpu).all():
            raise RuntimeError("Non-finite values (NaN/Inf) detected in VideoMAE output features.")
        return feats_cpu

    @torch.inference_mode()
    def extract_clips(
        self, clips_rgb: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> np.ndarray:
        """Extract clip-level VideoMAE representations from an array or sequence of RGB video clips.

        Args:
            clips_rgb: NumPy array or ClipWindowArray of shape `(num_clips, clip_len, H, W, 3)`, dtype uint8.

        Returns:
            NumPy array of shape `(num_clips, feature_dim)` with dtype float32.
        """
        if clips_rgb.ndim != 5 or clips_rgb.shape[-1] != 3:
            raise ValueError(
                f"Expected clips_rgb of shape (num_clips, T, H, W, 3), got {clips_rgb.shape}"
            )
        num_clips = clips_rgb.shape[0]
        if num_clips == 0:
            return np.empty((0, self.config.feature_dim), dtype=np.float32)

        batch_size = max(1, self.config.batch_size)
        all_features = []

        for start_idx in range(0, num_clips, batch_size):
            end_idx = min(num_clips, start_idx + batch_size)
            batch_tensors = [
                preprocess_clip_for_videomae(
                    clips_rgb[i],
                    num_frames=self.config.num_frames,
                    image_size=self.config.image_size,
                )
                for i in range(start_idx, end_idx)
            ]
            pixel_values = torch.stack(batch_tensors, dim=0)  # (B, T, 3, H, W)
            try:
                batch_feats = self.forward_batch(pixel_values)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and pixel_values.shape[0] > 1:
                    logger.warning(
                        "Out-of-memory during VideoMAE batch inference (B=%d); retrying clip-by-clip.",
                        pixel_values.shape[0],
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    single_feats = [
                        self.forward_batch(pixel_values[j : j + 1])
                        for j in range(pixel_values.shape[0])
                    ]
                    batch_feats = torch.cat(single_feats, dim=0)
                else:
                    raise
            all_features.append(batch_feats.numpy())

        result = np.concatenate(all_features, axis=0).astype(np.float32)
        if result.shape != (num_clips, self.config.feature_dim):
            raise RuntimeError(
                f"Unexpected VideoMAE output shape {result.shape}, expected {(num_clips, self.config.feature_dim)}"
            )
        return result
