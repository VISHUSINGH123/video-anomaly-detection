"""Data loading, preprocessing, annotation alignment, and feature caching package."""

from vad.data.annotations import (
    VideoAnnotation,
    align_annotations_to_windows,
    load_annotations_directory,
    load_annotations_from_json,
    parse_ucsd_matlab_annotations,
)
from vad.data.dataset import (
    CachedFeatureDataset,
    CachedVideoFeatures,
    collate_variable_length_features,
    load_cached_features,
    save_cached_features,
)
from vad.data.preprocessing import (
    preprocess_clip_for_videomae,
    resize_and_normalize_frames,
    sample_clip_frames_for_dinov2,
    uniform_temporal_subsample,
)
from vad.data.video_loader import (
    DecodedVideo,
    TemporalWindow,
    build_temporal_windows,
    decode_video,
    discover_videos,
)

__all__ = [
    "VideoAnnotation",
    "align_annotations_to_windows",
    "load_annotations_directory",
    "load_annotations_from_json",
    "parse_ucsd_matlab_annotations",
    "CachedFeatureDataset",
    "CachedVideoFeatures",
    "collate_variable_length_features",
    "load_cached_features",
    "save_cached_features",
    "preprocess_clip_for_videomae",
    "resize_and_normalize_frames",
    "sample_clip_frames_for_dinov2",
    "uniform_temporal_subsample",
    "DecodedVideo",
    "TemporalWindow",
    "build_temporal_windows",
    "decode_video",
    "discover_videos",
]
