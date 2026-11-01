"""Unit tests for frame sampling, clip construction, preprocessing, and encoder shape contracts."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from vad.config import DINOv2Config, VideoMAEConfig
from vad.data.preprocessing import (
    preprocess_clip_for_videomae,
    resize_and_normalize_frames,
    sample_clip_frames_for_dinov2,
    uniform_temporal_subsample,
)
from vad.data.video_loader import (
    ClipWindowArray,
    DecodedVideo,
    build_temporal_windows,
    decode_video,
    discover_videos,
)
from vad.models.dinov2_encoder import DINOv2Encoder
from vad.models.videomae_encoder import VideoMAEEncoder


class _DummyVideoMAEBackbone(nn.Module):
    """Lightweight mock backbone matching Hugging Face VideoMAEModel output signature `(B, N_tokens, D)`."""

    def __init__(self, hidden_size: int = 48) -> None:
        super().__init__()
        self.proj = nn.Linear(3, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        # pixel_values: (B, T, 3, H, W)
        b, t, c, h, w = pixel_values.shape
        spatial_pooled = pixel_values.mean(dim=(-1, -2))  # (B, T, 3)
        tokens = self.proj(spatial_pooled)  # (B, T, hidden_size)
        return SimpleNamespace(last_hidden_state=tokens)


class _DummyDINOv2Backbone(nn.Module):
    """Lightweight mock backbone matching Hugging Face Dinov2Model output signature `(N, 1+P, D)`."""

    def __init__(self, hidden_size: int = 48) -> None:
        super().__init__()
        self.proj = nn.Linear(3, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        # pixel_values: (N, 3, H, W)
        pooled = pixel_values.mean(dim=(-1, -2)).unsqueeze(1)  # (N, 1, 3)
        tokens = self.proj(pooled).repeat(1, 5, 1)  # (N, 5, hidden_size)
        return SimpleNamespace(last_hidden_state=tokens)


def _make_dummy_decoded_video(num_frames: int = 25, fps: float = 10.0) -> DecodedVideo:
    rng = np.random.default_rng(123)
    frames = rng.integers(0, 256, size=(num_frames, 64, 80, 3), dtype=np.uint8)
    indices = list(range(num_frames))
    timestamps = [i / fps for i in indices]
    return DecodedVideo(
        video_id="dummy_vid",
        source_path=Path("dummy_vid.mp4"),
        frames_rgb=frames,
        fps=fps,
        original_num_frames=num_frames,
        frame_indices=indices,
        timestamps_sec=timestamps,
    )


def test_resize_and_normalize_shapes_and_finite() -> None:
    rng = np.random.default_rng(42)
    frames = rng.integers(0, 256, size=(8, 120, 160, 3), dtype=np.uint8)
    tensor = resize_and_normalize_frames(frames, image_size=224)
    assert tensor.shape == (8, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_uniform_temporal_subsample() -> None:
    frames = np.zeros((20, 32, 32, 3), dtype=np.uint8)
    for i in range(20):
        frames[i] = i
    sampled = uniform_temporal_subsample(frames, num_samples=16)
    assert sampled.shape == (16, 32, 32, 3)
    assert sampled[0, 0, 0, 0] == 0
    assert sampled[-1, 0, 0, 0] == 19


def test_preprocess_clip_for_videomae_and_dinov2() -> None:
    rng = np.random.default_rng(7)
    clip = rng.integers(0, 256, size=(16, 100, 100, 3), dtype=np.uint8)

    v_tensor = preprocess_clip_for_videomae(clip, num_frames=16, image_size=224)
    assert v_tensor.shape == (16, 3, 224, 224)

    d_tensor, rel_indices = sample_clip_frames_for_dinov2(
        clip, frame_sample_rate=2, image_size=224
    )
    assert d_tensor.shape == (8, 3, 224, 224)
    assert rel_indices == [0, 2, 4, 6, 8, 10, 12, 14]


def test_build_temporal_windows_covering_tail_and_short_videos() -> None:
    decoded = _make_dummy_decoded_video(num_frames=25, fps=10.0)
    clips, windows = build_temporal_windows(decoded, clip_length=16, clip_stride=8)
    assert clips.shape == (3, 16, 64, 80, 3)
    assert len(windows) == 3
    assert windows[0].start_frame == 0 and windows[0].end_frame == 15
    assert windows[1].start_frame == 8 and windows[1].end_frame == 23
    # Tail coverage window
    assert windows[2].start_frame == 9 and windows[2].end_frame == 24
    assert not any(w.is_padded for w in windows)

    # Short video (< clip_length) with 'pad' policy
    short_vid = _make_dummy_decoded_video(num_frames=6, fps=10.0)
    padded_clips, padded_wins = build_temporal_windows(
        short_vid, clip_length=16, clip_stride=8, short_video_policy="pad"
    )
    assert padded_clips.shape == (1, 16, 64, 80, 3)
    assert len(padded_wins) == 1
    assert padded_wins[0].is_padded is True

    # Short video with 'drop' policy
    dropped_clips, dropped_wins = build_temporal_windows(
        short_vid, clip_length=16, clip_stride=8, short_video_policy="drop"
    )
    assert dropped_clips.shape[0] == 0
    assert dropped_wins == []


def test_decode_video_and_corrupt_file_handling(tmp_path: Path) -> None:
    vid_file = tmp_path / "valid_sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_file), fourcc, 15.0, (64, 64))
    for i in range(20):
        frame = np.full((64, 64, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    discovered = discover_videos(tmp_path)
    assert len(discovered) == 1

    decoded = decode_video(vid_file)
    assert decoded.frames_rgb.shape == (20, 64, 64, 3)
    assert len(decoded.timestamps_sec) == 20

    # Corrupt file handling
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"not a valid video stream")
    with pytest.raises(RuntimeError):
        decode_video(corrupt_file)


def test_videomae_and_dinov2_encoder_wrappers_without_network() -> None:
    """Verify VideoMAEEncoder and DINOv2Encoder using lightweight in-memory mock backbones."""
    tiny_videomae_backbone = _DummyVideoMAEBackbone(hidden_size=48)
    v_cfg = VideoMAEConfig(
        feature_dim=48,
        num_frames=4,
        image_size=32,
        batch_size=2,
        pooling="mean",
    )
    v_encoder = VideoMAEEncoder(config=v_cfg, backbone=tiny_videomae_backbone)

    rng = np.random.default_rng(99)
    clips_rgb = rng.integers(0, 256, size=(3, 4, 32, 32, 3), dtype=np.uint8)
    v_feats = v_encoder.extract_clips(clips_rgb)
    assert v_feats.shape == (3, 48)
    assert np.isfinite(v_feats).all()

    tiny_dinov2_backbone = _DummyDINOv2Backbone(hidden_size=48)
    d_cfg = DINOv2Config(
        feature_dim=48,
        image_size=32,
        frame_sample_rate=2,
        batch_size=4,
        pooling="cls_and_std",
    )
    d_encoder = DINOv2Encoder(config=d_cfg, backbone=tiny_dinov2_backbone)
    d_feats, sampled_idx = d_encoder.extract_clips(clips_rgb)
    assert d_feats.shape == (3, 48)
    assert len(sampled_idx) == 3
    assert sampled_idx[0] == [0, 2]
    assert np.isfinite(d_feats).all()


def test_high_res_memory_efficient_windowing_and_decode(tmp_path: Path) -> None:
    """Verify zero-copy ClipWindowArray and max_dimension video downscaling for HD surveillance inputs."""
    # 1. Test video decode with max_dimension constraint
    vid_file = tmp_path / "hd_sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Write a 128x256 test video
    writer = cv2.VideoWriter(str(vid_file), fourcc, 10.0, (256, 128))
    for i in range(20):
        frame = np.full((128, 256, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # Without max_dimension: retains 128x256
    native_decoded = decode_video(vid_file)
    assert native_decoded.frames_rgb.shape == (20, 128, 256, 3)

    # With max_dimension=64: scaled down so max dimension is 64
    scaled_decoded = decode_video(vid_file, max_dimension=64)
    assert scaled_decoded.frames_rgb.shape[1] == 32
    assert scaled_decoded.frames_rgb.shape[2] == 64

    # 2. Test memory-efficient fallback in build_temporal_windows
    # Set max_dense_stack_bytes threshold small (1 KB) to trigger ClipWindowArray
    clips, windows = build_temporal_windows(
        native_decoded, clip_length=16, clip_stride=2, max_dense_stack_bytes=1024
    )
    assert isinstance(clips, ClipWindowArray)
    assert clips.ndim == 5
    assert len(clips) == len(windows)
    assert clips.shape == (len(windows), 16, 128, 256, 3)
    # Check indexing and zero-copy slicing
    assert clips[0].shape == (16, 128, 256, 3)
    assert clips[1:3].shape == (2, 16, 128, 256, 3)
    assert isinstance(clips[1:3], ClipWindowArray)
    # Check iteration
    count = sum(1 for _ in clips)
    assert count == len(windows)
    # Check __array__ conversion
    arr = np.asarray(clips[:2])
    assert arr.shape == (2, 16, 128, 256, 3)
