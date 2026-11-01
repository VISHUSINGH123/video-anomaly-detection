"""Logging utilities for consistent console and file formatting."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "vad",
    level: Optional[str] = None,
    log_file: Optional[Path | str] = None,
) -> logging.Logger:
    """Configure and return a structured logger.

    Args:
        name: Logger name.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR). Defaults to VAD_LOG_LEVEL or INFO.
        log_file: Optional file path to also write logs.

    Returns:
        Configured logging.Logger instance.
    """
    resolved_level = (level or os.environ.get("VAD_LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
