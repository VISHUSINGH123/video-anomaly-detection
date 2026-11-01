"""Video discovery, decoding, frame sequence loading, and temporal clip construction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from vad.utils.logging import setup_logger

logger = setup_logger("vad.video_loader")

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}
SUPPORTED_IMAGE_EXTENSIONS = {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg"}


class ClipWindowArray(Sequence[np.ndarray]):
    """Memory-efficient sequence of video clips backed by sliced views of decoded frames.

    Provides standard array-like attributes (.shape, .ndim, .dtype) and index access (.clips[i]),
    avoiding eager 5D contiguous array materialization to prevent multi-gigabyte OOM allocations.
    """

    def __init__(
        self,
        clips: List[np.ndarray],
        shape: Tuple[int, int, int, int, int],
        dtype: np.dtype | type = np.uint8,
    ) -> None:
        self._clips = clips
        self.shape = shape
        self.ndim = 5
        self.dtype = np.dtype(dtype)

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, (int, np.integer)):
            return self._clips[idx]
        elif isinstance(idx, slice):
            sub = self._clips[idx]
            new_shape = (len(sub),) + self.shape[1:]
            return ClipWindowArray(sub, new_shape, self.dtype)
        elif isinstance(idx, tuple):
            first = idx[0]
            rest = idx[1:]
            if isinstance(first, (int, np.integer)):
                return self._clips[first][rest]
            elif isinstance(first, slice):
                sub = self._clips[first]
                new_shape = (len(sub),) + self.shape[1:]
                return ClipWindowArray([c[rest] for c in sub], new_shape, self.dtype)
        raise TypeError(f"Unsupported index type {type(idx)} for ClipWindowArray")

    def __iter__(self):
        return iter(self._clips)

    def __array__(self, dtype=None) -> np.ndarray:
        arr = np.stack(self._clips, axis=0)
        return arr if dtype is None else arr.astype(dtype)

    @property
    def nbytes(self) -> int:
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod * self.dtype.itemsize


@dataclass(frozen=True)
class TemporalWindow:
    """Metadata and frame indices for a single fixed-length temporal window (clip)."""

    clip_index: int
    frame_indices: List[int]
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    center_time_sec: float
    is_padded: bool


@dataclass
class DecodedVideo:
    """Container for decoded RGB frames and temporal metadata."""

    video_id: str
    source_path: Path
    frames_rgb: np.ndarray  # Shape: (N, H, W, 3), dtype=uint8
    fps: float
    original_num_frames: int
    frame_indices: List[int]
    timestamps_sec: List[float]


def discover_videos(root_dir: Path | str) -> List[Path]:
    """Discover video files or frame-sequence directories (e.g. UCSD Ped1/Ped2).

    Args:
        root_dir: Directory to scan recursively.

    Returns:
        Sorted list of Paths pointing either to video files or frame-sequence folders.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Video directory does not exist: {root}")

    if root.is_file():
        if root.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            return [root]
        raise ValueError(f"Unsupported video file extension: {root.suffix}")

    discovered: List[Path] = []

    # 1. Standard video files
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            discovered.append(p)

    # 2. Frame sequence directories (e.g., UCSD Ped2 Test001/*.tif)
    for d in sorted([p for p in root.rglob("*") if p.is_dir()] + [root]):
        if d.name.endswith("_gt"):
            continue  # Skip UCSD pixel-mask ground truth folders
        images = [
            f
            for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if len(images) >= 2:
            discovered.append(d)

    # Deduplicate while preserving deterministic order
    seen = set()
    unique: List[Path] = []
    for item in discovered:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(item)
    return unique


def _decode_image_directory(
    dir_path: Path, default_fps: float = 25.0, max_dimension: Optional[int] = None
) -> DecodedVideo:
    """Decode a directory of sequential image frames (such as UCSD Pedestrian .tif folders)."""
    image_files = sorted(
        [
            f
            for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
    )
    if not image_files:
        raise ValueError(f"No supported image frames found in directory: {dir_path}")

    frames: List[np.ndarray] = []
    for img_path in image_files:
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            if max_dimension is not None:
                w, h = rgb_img.size
                max_dim = max(w, h)
                if max_dim > max_dimension:
                    scale = max_dimension / float(max_dim)
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    rgb_img = rgb_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            frames.append(np.asarray(rgb_img, dtype=np.uint8))

    stacked = np.stack(frames, axis=0)
    num_frames = stacked.shape[0]
    indices = list(range(num_frames))
    timestamps = [idx / default_fps for idx in indices]
    return DecodedVideo(
        video_id=dir_path.name,
        source_path=dir_path,
        frames_rgb=stacked,
        fps=default_fps,
        original_num_frames=num_frames,
        frame_indices=indices,
        timestamps_sec=timestamps,
    )


def decode_video(
    video_path: Path | str,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    max_dimension: Optional[int] = None,
) -> DecodedVideo:
    """Decode a video file or frame-sequence directory into RGB uint8 frames.

    Args:
        video_path: Path to a video file or directory of frame images.
        target_fps: Optional target frame rate for temporal subsampling.
        max_frames: Optional cap on the maximum number of decoded frames.
        max_dimension: Optional maximum spatial dimension (width or height) to preserve memory on HD/4K videos.

    Returns:
        DecodedVideo with RGB frames (N, H, W, 3) and exact frame indices/timestamps.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video path not found: {path}")

    if path.is_dir():
        decoded = _decode_image_directory(
            path, default_fps=target_fps or 25.0, max_dimension=max_dimension
        )
        if max_frames is not None and decoded.frames_rgb.shape[0] > max_frames:
            decoded.frames_rgb = decoded.frames_rgb[:max_frames]
            decoded.frame_indices = decoded.frame_indices[:max_frames]
            decoded.timestamps_sec = decoded.timestamps_sec[:max_frames]
        return decoded

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{path.suffix}' for {path.name}. "
            f"Supported extensions: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open or decode video file (corrupt or unsupported): {path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(native_fps) or native_fps <= 0.1:
        native_fps = 25.0

    stride = 1
    if target_fps is not None and target_fps > 0 and target_fps < native_fps:
        stride = max(1, int(round(native_fps / target_fps)))

    frames_rgb: List[np.ndarray] = []
    frame_indices: List[int] = []
    timestamps_sec: List[float] = []

    raw_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            if raw_idx % stride == 0:
                if max_dimension is not None:
                    h, w = frame_bgr.shape[:2]
                    max_dim = max(h, w)
                    if max_dim > max_dimension:
                        scale = max_dimension / float(max_dim)
                        new_w = max(1, int(round(w * scale)))
                        new_h = max(1, int(round(h * scale)))
                        frame_bgr = cv2.resize(
                            frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
                        )
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)
                frame_indices.append(raw_idx)
                timestamps_sec.append(raw_idx / native_fps)
                if max_frames is not None and len(frames_rgb) >= max_frames:
                    break
            raw_idx += 1
    finally:
        cap.release()

    if len(frames_rgb) == 0:
        raise RuntimeError(f"Video file contained zero readable frames: {path}")

    effective_fps = native_fps / stride
    return DecodedVideo(
        video_id=path.stem,
        source_path=path,
        frames_rgb=np.stack(frames_rgb, axis=0),
        fps=effective_fps,
        original_num_frames=raw_idx,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
    )


def build_temporal_windows(
    decoded: DecodedVideo,
    clip_length: int = 16,
    clip_stride: int = 8,
    short_video_policy: str = "pad",
    max_dense_stack_bytes: int = 128 * 1024 * 1024,
) -> Tuple[Union[np.ndarray, ClipWindowArray], List[TemporalWindow]]:
    """Construct fixed-length temporal clips from a DecodedVideo while preserving timestamps.

    Args:
        decoded: DecodedVideo object containing frames and timestamps.
        clip_length: Number of frames per temporal window.
        clip_stride: Step size in frames between consecutive windows.
        short_video_policy: 'pad' (repeat last frame when video < clip_length) or 'drop'.
        max_dense_stack_bytes: Maximum memory threshold (bytes) before falling back to zero-copy
            ClipWindowArray instead of allocating a monolithic 5D array.

    Returns:
        Tuple of:
          - clips_rgb: np.ndarray or ClipWindowArray of shape (num_clips, clip_length, H, W, 3)
          - windows: List of TemporalWindow metadata objects aligned with clips_rgb
    """
    if clip_length < 2:
        raise ValueError(f"clip_length must be >= 2, got {clip_length}")
    if clip_stride < 1:
        raise ValueError(f"clip_stride must be >= 1, got {clip_stride}")

    total_sampled = decoded.frames_rgb.shape[0]
    if total_sampled == 0:
        raise ValueError(f"DecodedVideo '{decoded.video_id}' has 0 frames.")

    if total_sampled < clip_length:
        if short_video_policy == "drop":
            return (
                np.empty((0, clip_length, *decoded.frames_rgb.shape[1:]), dtype=np.uint8),
                [],
            )
        elif short_video_policy == "pad":
            pad_count = clip_length - total_sampled
            pad_frames = np.repeat(decoded.frames_rgb[-1:], pad_count, axis=0)
            clip_frames = np.concatenate([decoded.frames_rgb, pad_frames], axis=0)
            orig_indices = decoded.frame_indices + [decoded.frame_indices[-1]] * pad_count
            start_t = decoded.timestamps_sec[0]
            end_t = decoded.timestamps_sec[-1]
            win = TemporalWindow(
                clip_index=0,
                frame_indices=orig_indices,
                start_frame=orig_indices[0],
                end_frame=orig_indices[-1],
                start_time_sec=float(start_t),
                end_time_sec=float(end_t),
                center_time_sec=float(0.5 * (start_t + end_t)),
                is_padded=True,
            )
            return np.expand_dims(clip_frames, axis=0), [win]
        else:
            raise ValueError(f"Invalid short_video_policy: {short_video_policy}")

    starts = list(range(0, total_sampled - clip_length + 1, clip_stride))
    last_covered = starts[-1] + clip_length - 1
    # Ensure tail frames are covered by adding one final window if needed
    if last_covered < total_sampled - 1:
        starts.append(total_sampled - clip_length)

    clips_list: List[np.ndarray] = []
    windows: List[TemporalWindow] = []

    for clip_idx, s in enumerate(starts):
        e = s + clip_length
        sub_frames = decoded.frames_rgb[s:e]
        sub_indices = decoded.frame_indices[s:e]
        sub_times = decoded.timestamps_sec[s:e]
        clips_list.append(sub_frames)
        windows.append(
            TemporalWindow(
                clip_index=clip_idx,
                frame_indices=list(sub_indices),
                start_frame=int(sub_indices[0]),
                end_frame=int(sub_indices[-1]),
                start_time_sec=float(sub_times[0]),
                end_time_sec=float(sub_times[-1]),
                center_time_sec=float(0.5 * (sub_times[0] + sub_times[-1])),
                is_padded=False,
            )
        )

    num_clips = len(clips_list)
    h, w, c = decoded.frames_rgb.shape[1:]
    elem_bytes = decoded.frames_rgb.dtype.itemsize
    dense_bytes = num_clips * clip_length * h * w * c * elem_bytes

    if dense_bytes > max_dense_stack_bytes:
        logger.info(
            "Large sliding window requested (%d clips of %dx%dx%dx%d = %.2f GiB dense array). "
            "Using zero-copy ClipWindowArray to preserve memory.",
            num_clips,
            clip_length,
            h,
            w,
            c,
            dense_bytes / (1024**3),
        )
        return ClipWindowArray(
            clips_list,
            shape=(num_clips, clip_length, h, w, c),
            dtype=decoded.frames_rgb.dtype,
        ), windows

    return np.stack(clips_list, axis=0), windows
