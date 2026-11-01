"""Deterministic seeding and environment metadata capture for reproducibility."""

import os
import platform
import random
import sys
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer random seed.
        deterministic_cudnn: Whether to configure deterministic cuDNN behavior when CUDA is present.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_reproducibility_metadata(seed: int) -> Dict[str, Any]:
    """Capture hardware and software versions for experiment tracking and checkpointing."""
    meta: Dict[str, Any] = {
        "seed": seed,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta["cuda_version"] = torch.version.cuda
        meta["gpu_name"] = torch.cuda.get_device_name(0)
    return meta
