"""Configuration schemas and YAML loading/validation for the VAD pipeline."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    dataset_name: str = "ucsd_ped2"
    raw_video_dir: str = "data/raw"
    processed_feature_dir: str = "data/processed"
    annotations_dir: str = "data/annotations"
    clip_length: int = 16
    clip_stride: int = 8
    target_fps: Optional[float] = None
    short_video_policy: str = "pad"  # "pad" or "drop"
    max_dimension: Optional[int] = 384

    def validate(self) -> None:
        if self.clip_length < 2:
            raise ValueError(f"clip_length must be >= 2, got {self.clip_length}")
        if self.clip_stride < 1:
            raise ValueError(f"clip_stride must be >= 1, got {self.clip_stride}")
        if self.short_video_policy not in ("pad", "drop"):
            raise ValueError(
                f"short_video_policy must be 'pad' or 'drop', got '{self.short_video_policy}'"
            )
        if self.max_dimension is not None and self.max_dimension < 64:
            raise ValueError(
                f"max_dimension must be >= 64, got {self.max_dimension}"
            )


@dataclass
class VideoMAEConfig:
    enabled: bool = True
    model_name: str = "MCG-NJU/videomae-base"
    feature_dim: int = 768
    num_frames: int = 16
    image_size: int = 224
    batch_size: int = 4
    pooling: str = "mean"
    use_fp16: bool = False
    local_files_only: bool = False


@dataclass
class DINOv2Config:
    enabled: bool = True
    model_name: str = "facebook/dinov2-base"
    feature_dim: int = 768
    image_size: int = 224
    frame_sample_rate: int = 2
    batch_size: int = 16
    pooling: str = "cls_and_std"  # "cls" or "cls_and_std"
    use_fp16: bool = False
    local_files_only: bool = False


@dataclass
class EncodersConfig:
    videomae: VideoMAEConfig = field(default_factory=VideoMAEConfig)
    dinov2: DINOv2Config = field(default_factory=DINOv2Config)


@dataclass
class TemporalConfig:
    enabled: bool = True
    hidden_dim: int = 256
    num_layers: int = 4
    kernel_size: int = 3
    dilation_base: int = 2
    dropout: float = 0.2
    causal: bool = False

    def validate(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError(f"temporal.hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.num_layers < 1:
            raise ValueError(f"temporal.num_layers must be >= 1, got {self.num_layers}")
        if self.kernel_size < 2 or self.kernel_size % 2 == 0:
            raise ValueError(
                f"temporal.kernel_size must be an odd integer >= 3, got {self.kernel_size}"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"temporal.dropout must be in [0, 1), got {self.dropout}")


@dataclass
class FusionConfig:
    mode: str = "gated_multiprior"  # gated_multiprior, concat_mlp, videomae_only, dinov2_only, no_tcn
    proj_dim: int = 256
    dropout: float = 0.2
    include_motion_prior: bool = True
    temperature: float = 1.0

    def validate(self) -> None:
        valid_modes = {
            "gated_multiprior",
            "concat_mlp",
            "videomae_only",
            "dinov2_only",
            "no_tcn",
        }
        if self.mode not in valid_modes:
            raise ValueError(f"fusion.mode must be one of {sorted(valid_modes)}, got '{self.mode}'")
        if self.proj_dim < 1:
            raise ValueError(f"fusion.proj_dim must be >= 1, got {self.proj_dim}")
        if self.temperature <= 0.0:
            raise ValueError(f"fusion.temperature must be > 0, got {self.temperature}")


@dataclass
class TrainingConfig:
    supervision: str = "supervised_clip"  # supervised_clip, weakly_supervised_mil
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    pos_weight: float = 2.0
    smoothness_lambda: float = 0.05
    sparsity_lambda: float = 0.005
    mil_topk_ratio: float = 0.15
    early_stopping_patience: int = 8
    checkpoint_dir: str = "checkpoints"

    def validate(self) -> None:
        if self.supervision not in ("supervised_clip", "weakly_supervised_mil"):
            raise ValueError(
                "training.supervision must be 'supervised_clip' or 'weakly_supervised_mil', "
                f"got '{self.supervision}'"
            )
        if self.epochs < 1:
            raise ValueError(f"training.epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"training.batch_size must be >= 1, got {self.batch_size}")
        if not (0.0 < self.mil_topk_ratio <= 1.0):
            raise ValueError(f"training.mil_topk_ratio must be in (0, 1], got {self.mil_topk_ratio}")


@dataclass
class EvaluationConfig:
    threshold: float = 0.5
    smoothing_window: int = 3
    output_dir: str = "outputs/evaluation"


@dataclass
class InferenceConfig:
    window_length: int = 16
    window_stride: int = 8
    smoothing_window: int = 3
    threshold: float = 0.5
    output_dir: str = "outputs/predictions"


@dataclass
class VADConfig:
    experiment_name: str = "vad_multiprior_default"
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    encoders: EncodersConfig = field(default_factory=EncodersConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def validate(self) -> None:
        self.data.validate()
        self.temporal.validate()
        self.fusion.validate()
        self.training.validate()
        if not self.encoders.videomae.enabled and not self.encoders.dinov2.enabled:
            raise ValueError("At least one encoder (videomae or dinov2) must be enabled.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Optional[Path | str] = None, overrides: Optional[Dict[str, Any]] = None) -> VADConfig:
    """Load VADConfig from a YAML file and apply optional dictionary overrides."""
    raw: Dict[str, Any] = {}
    if path is not None:
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                raw = loaded

    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(raw.get(k), dict):
                raw[k].update(v)
            else:
                raw[k] = v

    data_cfg = DataConfig(**raw.get("data", {}))
    enc_raw = raw.get("encoders", {})
    encoders_cfg = EncodersConfig(
        videomae=VideoMAEConfig(**enc_raw.get("videomae", {})),
        dinov2=DINOv2Config(**enc_raw.get("dinov2", {})),
    )
    temporal_cfg = TemporalConfig(**raw.get("temporal", {}))
    fusion_cfg = FusionConfig(**raw.get("fusion", {}))
    training_cfg = TrainingConfig(**raw.get("training", {}))
    evaluation_cfg = EvaluationConfig(**raw.get("evaluation", {}))
    inference_cfg = InferenceConfig(**raw.get("inference", {}))

    cfg = VADConfig(
        experiment_name=raw.get("experiment_name", "vad_multiprior_default"),
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        data=data_cfg,
        encoders=encoders_cfg,
        temporal=temporal_cfg,
        fusion=fusion_cfg,
        training=training_cfg,
        evaluation=evaluation_cfg,
        inference=inference_cfg,
    )
    cfg.validate()
    return cfg
