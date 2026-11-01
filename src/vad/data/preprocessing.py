"""Deterministic frame and clip preprocessing for VideoMAE and DINOv2 encoders."""

from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_and_normalize_frames(
    frames_rgb: np.ndarray,
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Resize RGB uint8 frames and normalize with ImageNet statistics.

    Args:
        frames_rgb: NumPy array of shape (T, H, W, 3) with dtype uint8 in [0, 255].
        image_size: Target spatial height and width.
        mean: Per-channel RGB mean.
        std: Per-channel RGB standard deviation.

    Returns:
        Float32 tensor of shape (T, 3, image_size, image_size).
    """
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(
            f"Expected frames_rgb of shape (T, H, W, 3), got {frames_rgb.shape}"
        )
    if frames_rgb.shape[0] == 0:
        raise ValueError("Cannot preprocess an empty sequence of frames (T=0).")

    resized_list = []
    for i in range(frames_rgb.shape[0]):
        frame = frames_rgb[i]
        if frame.shape[0] != image_size or frame.shape[1] != image_size:
            frame = cv2.resize(
                frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR
            )
        resized_list.append(frame)

    arr = np.stack(resized_list, axis=0).astype(np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, 3)
    norm_arr = (arr - mean_arr) / std_arr

    # Transpose (T, H, W, C) -> (T, C, H, W)
    tensor = torch.from_numpy(np.ascontiguousarray(norm_arr.transpose(0, 3, 1, 2)))
    return tensor


def uniform_temporal_subsample(frames_rgb: np.ndarray, num_samples: int) -> np.ndarray:
    """Uniformly sample `num_samples` frames along the temporal axis of `frames_rgb`.

    Args:
        frames_rgb: Array of shape (T, H, W, 3).
        num_samples: Desired number of output frames.

    Returns:
        Array of shape (num_samples, H, W, 3).
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    total = frames_rgb.shape[0]
    if total == 0:
        raise ValueError("Cannot subsample empty frame array.")
    if total == num_samples:
        return frames_rgb
    indices = np.linspace(0, total - 1, num=num_samples)
    indices = np.clip(np.round(indices).astype(np.int64), 0, total - 1)
    return frames_rgb[indices]


def preprocess_clip_for_videomae(
    clip_rgb: np.ndarray,
    num_frames: int = 16,
    image_size: int = 224,
) -> torch.Tensor:
    """Prepare a single clip for Hugging Face VideoMAEModel.

    VideoMAE expects input `pixel_values` of shape `(B, T, C, H, W)`.

    Args:
        clip_rgb: NumPy array of shape (T, H, W, 3), dtype uint8.
        num_frames: Number of frames required by the VideoMAE position embeddings (default 16).
        image_size: Spatial resolution (default 224).

    Returns:
        Float32 tensor of shape (num_frames, 3, image_size, image_size).
    """
    sampled = uniform_temporal_subsample(clip_rgb, num_samples=num_frames)
    return resize_and_normalize_frames(sampled, image_size=image_size)


def sample_clip_frames_for_dinov2(
    clip_rgb: np.ndarray,
    frame_sample_rate: int = 2,
    image_size: int = 224,
) -> Tuple[torch.Tensor, List[int]]:
    """Sample frames from a clip at `frame_sample_rate` and preprocess for DINOv2.

    Args:
        clip_rgb: NumPy array of shape (T, H, W, 3), dtype uint8.
        frame_sample_rate: Stride within the clip (e.g., 2 samples 8 frames from a 16-frame clip).
        image_size: Spatial resolution (default 224).

    Returns:
        Tuple of:
          - Float32 tensor of shape (K, 3, image_size, image_size)
          - List of relative indices within the clip [0, ..., T-1]
    """
    if frame_sample_rate < 1:
        raise ValueError(f"frame_sample_rate must be >= 1, got {frame_sample_rate}")
    total = clip_rgb.shape[0]
    rel_indices = list(range(0, total, frame_sample_rate))
    if not rel_indices:
        rel_indices = [0]
    sampled = clip_rgb[rel_indices]
    tensor = resize_and_normalize_frames(sampled, image_size=image_size)
    return tensor, rel_indices
