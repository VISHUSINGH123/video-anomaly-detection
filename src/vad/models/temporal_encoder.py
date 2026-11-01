"""Dilated Residual Temporal Convolutional Network (TCN) for sequence context modeling.

Architecture notes:
- Supports both **non-causal** (symmetric padding, bidirectional temporal context for offline or
  buffered window scoring) and **strictly causal** (left-only padding, zero future leakage for
  online frame-by-frame streaming) modes via `causal: bool`.
- Applies explicit boolean sequence masking `(B, T)` before and after every temporal convolution
  layer so padded positions in variable-length video batches never leak into valid clip features.
- Receptive field for `L` layers with kernel size `K` and dilation base `b` (two convs per block):
  `R = 1 + 2 * (K - 1) * sum_{l=0}^{L-1} b^l`.
  For default `L=4, K=3, b=2`: `R = 1 + 4 * (1 + 2 + 4 + 8) = 61` clips.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChannelLayerNorm1d(nn.Module):
    """Per-timestep LayerNorm over the channel axis of a `(B, C, T)` tensor (strictly causal-safe)."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C) -> LayerNorm(C) -> (B, C, T)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalConvBlock(nn.Module):
    """Single dilated residual 1D convolutional block with causal/non-causal padding and masking."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size < 2 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd integer >= 3, got {kernel_size}")

        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.total_pad = (kernel_size - 1) * dilation

        # When non-causal, symmetric padding is applied directly in Conv1d;
        # when causal, we pad manually on the left and trim no future frames.
        conv_pad = 0 if causal else (self.total_pad // 2)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=conv_pad,
        )
        self.norm1 = _ChannelLayerNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=conv_pad,
        )
        self.norm2 = _ChannelLayerNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample: Optional[nn.Module] = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def _apply_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        """Apply 1D convolution with causal left-padding or symmetric padding."""
        if self.causal:
            x_padded = F.pad(x, (self.total_pad, 0))
            return conv(x_padded)
        return conv(x)

    @staticmethod
    def _apply_mask(x: torch.Tensor, mask_b1t: Optional[torch.Tensor]) -> torch.Tensor:
        if mask_b1t is not None:
            return x * mask_b1t
        return x

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape `(B, C_in, T)`.
            mask: Optional boolean validity mask of shape `(B, T)` where True = valid timestep.

        Returns:
            Tensor of shape `(B, C_out, T)`.
        """
        mask_b1t = mask.unsqueeze(1).to(x.dtype) if mask is not None else None
        x = self._apply_mask(x, mask_b1t)

        residual = self.downsample(x) if self.downsample is not None else x

        out = self._apply_conv(self.conv1, x)
        out = self._apply_mask(out, mask_b1t)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)
        out = self._apply_mask(out, mask_b1t)

        out = self._apply_conv(self.conv2, out)
        out = self._apply_mask(out, mask_b1t)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)

        out = out + residual
        return self._apply_mask(out, mask_b1t)


class TemporalConvNet(nn.Module):
    """Multi-layer dilated Temporal Convolutional Network over clip sequences."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.2,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if in_dim < 1 or hidden_dim < 1 or num_layers < 1:
            raise ValueError("in_dim, hidden_dim, and num_layers must all be >= 1.")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.causal = causal

        blocks = []
        for layer_idx in range(num_layers):
            dilation = int(dilation_base**layer_idx)
            c_in = in_dim if layer_idx == 0 else hidden_dim
            blocks.append(
                TemporalConvBlock(
                    in_channels=c_in,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    causal=causal,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(hidden_dim)

    @property
    def receptive_field(self) -> int:
        """Compute exact temporal receptive field in number of clips."""
        total = 1
        for i in range(self.num_layers):
            d = int(self.dilation_base**i)
            total += 2 * (self.kernel_size - 1) * d
        return total

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process a sequence of clip features `(B, T, D_in)` and return `(B, T, hidden_dim)`.

        Args:
            x: Input sequence tensor of shape `(B, T, D_in)`.
            mask: Optional boolean mask of shape `(B, T)` where True = valid clip.

        Returns:
            Temporally modeled sequence tensor of shape `(B, T, hidden_dim)`.
        """
        if x.ndim != 3:
            raise ValueError(f"TCN expects 3D input (B, T, D), got {tuple(x.shape)}")
        if mask is not None and mask.shape != x.shape[:2]:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match input (B, T) {tuple(x.shape[:2])}"
            )

        # Transpose (B, T, D) -> (B, D, T) for Conv1d
        h = x.transpose(1, 2)
        for block in self.blocks:
            h = block(h, mask=mask)
        # Transpose back (B, hidden_dim, T) -> (B, T, hidden_dim)
        out = h.transpose(1, 2)
        out = self.out_norm(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out
