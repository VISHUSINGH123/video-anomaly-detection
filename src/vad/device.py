"""Hardware device detection supporting CPU, NVIDIA CUDA, Apple MPS, and AMD/Windows DirectML."""

import os
import platform
from dataclasses import dataclass
from typing import Optional

import torch

from vad.utils.logging import setup_logger

logger = setup_logger("vad.device")


@dataclass(frozen=True)
class DeviceInfo:
    """Structured report of the resolved PyTorch compute device."""

    device: torch.device
    device_type: str
    device_name: str
    cuda_available: bool
    mps_available: bool
    directml_available: bool
    cpu_processor: str
    supports_fp16: bool


def _is_directml_available() -> bool:
    """Check if torch-directml is installed and functional (useful for AMD GPUs on Windows)."""
    try:
        import torch_directml  # type: ignore

        return bool(torch_directml.is_available())
    except Exception:
        return False


def resolve_device(requested: Optional[str] = None) -> DeviceInfo:
    """Resolve the execution device safely without assuming an NVIDIA GPU exists.

    Args:
        requested: Optional device string ("auto", "cpu", "cuda", "directml", "mps").
                   Defaults to VAD_DEVICE environment variable or "auto".

    Returns:
        DeviceInfo describing the selected device and hardware capabilities.
    """
    req = (requested or os.environ.get("VAD_DEVICE", "auto")).strip().lower()
    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    dml_ok = _is_directml_available()
    cpu_proc = platform.processor() or platform.machine() or "CPU"

    if req == "auto":
        if cuda_ok:
            dev = torch.device("cuda")
            dev_type = "cuda"
            dev_name = torch.cuda.get_device_name(0)
            fp16 = True
        elif dml_ok:
            import torch_directml  # type: ignore

            dev = torch_directml.device()
            dev_type = "directml"
            dev_name = str(torch_directml.device_name(0))
            fp16 = False
        elif mps_ok:
            dev = torch.device("mps")
            dev_type = "mps"
            dev_name = "Apple Silicon MPS"
            fp16 = False
        else:
            dev = torch.device("cpu")
            dev_type = "cpu"
            dev_name = f"CPU ({cpu_proc})"
            fp16 = False
    elif req.startswith("cuda"):
        if not cuda_ok:
            logger.warning(
                "Requested device '%s' but CUDA is not available on this machine (%s). "
                "Falling back to CPU execution.",
                req,
                cpu_proc,
            )
            dev = torch.device("cpu")
            dev_type = "cpu"
            dev_name = f"CPU ({cpu_proc})"
            fp16 = False
        else:
            dev = torch.device(req)
            dev_type = "cuda"
            dev_name = torch.cuda.get_device_name(dev)
            fp16 = True
    elif req == "directml":
        if not dml_ok:
            logger.warning(
                "Requested 'directml' but torch-directml is not installed or unavailable. "
                "Falling back to CPU execution."
            )
            dev = torch.device("cpu")
            dev_type = "cpu"
            dev_name = f"CPU ({cpu_proc})"
            fp16 = False
        else:
            import torch_directml  # type: ignore

            dev = torch_directml.device()
            dev_type = "directml"
            dev_name = str(torch_directml.device_name(0))
            fp16 = False
    elif req == "mps":
        if not mps_ok:
            logger.warning("Requested 'mps' but MPS is unavailable. Falling back to CPU.")
            dev = torch.device("cpu")
            dev_type = "cpu"
            dev_name = f"CPU ({cpu_proc})"
            fp16 = False
        else:
            dev = torch.device("mps")
            dev_type = "mps"
            dev_name = "Apple Silicon MPS"
            fp16 = False
    elif req == "cpu":
        dev = torch.device("cpu")
        dev_type = "cpu"
        dev_name = f"CPU ({cpu_proc})"
        fp16 = False
    else:
        raise ValueError(
            f"Unsupported device specification '{requested}'. "
            "Valid options are: 'auto', 'cpu', 'cuda', 'directml', 'mps'."
        )

    return DeviceInfo(
        device=dev,
        device_type=dev_type,
        device_name=dev_name,
        cuda_available=cuda_ok,
        mps_available=mps_ok,
        directml_available=dml_ok,
        cpu_processor=cpu_proc,
        supports_fp16=fp16,
    )
