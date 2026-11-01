"""Checkpoint persistence with full model config, optimizer state, and reproducibility metadata."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from vad.config import VADConfig, load_config
from vad.models.anomaly_detector import AnomalyDetector
from vad.utils.reproducibility import get_reproducibility_metadata


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    config: VADConfig,
    epoch: int,
    metrics: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    is_synthetic: bool = False,
) -> Path:
    """Save a complete training checkpoint to disk.

    Args:
        path: Target `.pt` file path.
        model: Trained AnomalyDetector instance.
        config: VADConfig used for training.
        epoch: Current epoch number.
        metrics: Validation/training metrics dictionary.
        optimizer: Optional PyTorch optimizer.
        is_synthetic: Flag explicitly recording whether the checkpoint was trained on synthetic smoke data.

    Returns:
        Resolved Path to the saved checkpoint.
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "config_dict": config.to_dict(),
        "metrics": metrics,
        "is_synthetic": bool(is_synthetic),
        "reproducibility": get_reproducibility_metadata(config.seed),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, ckpt_path)
    return ckpt_path


def load_checkpoint(
    path: Path | str,
    device: Optional[torch.device] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[AnomalyDetector, VADConfig, Dict[str, Any]]:
    """Load an AnomalyDetector and its VADConfig from a saved checkpoint.

    Args:
        path: Path to `.pt` checkpoint file.
        device: Target compute device to map weights onto.
        optimizer: Optional optimizer to restore state into.

    Returns:
        Tuple of `(model, config, checkpoint_metadata)`.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    map_loc = device or torch.device("cpu")
    payload = torch.load(ckpt_path, map_location=map_loc, weights_only=False)

    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid or corrupted checkpoint format in {ckpt_path}")

    cfg_dict = payload.get("config_dict", {})
    config = load_config(path=None, overrides=cfg_dict)

    model = AnomalyDetector(config=config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_loc)

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    meta = {
        "epoch": payload.get("epoch", 0),
        "metrics": payload.get("metrics", {}),
        "is_synthetic": bool(payload.get("is_synthetic", False)),
        "reproducibility": payload.get("reproducibility", {}),
    }
    return model, config, meta
