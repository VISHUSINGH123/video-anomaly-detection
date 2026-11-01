"""Pre-cache pretrained Hugging Face models for VideoMAE and DINOv2."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transformers import AutoImageProcessor, AutoModel, VideoMAEImageProcessor, VideoMAEModel
from vad.utils.logging import setup_logger

logger = setup_logger("vad.cache_models")


def main() -> int:
    logger.info("Downloading and caching VideoMAE (MCG-NJU/videomae-base)...")
    VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    logger.info("VideoMAE cached successfully.")

    logger.info("Downloading and caching DINOv2 (facebook/dinov2-base)...")
    AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    AutoModel.from_pretrained("facebook/dinov2-base")
    logger.info("DINOv2 cached successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
