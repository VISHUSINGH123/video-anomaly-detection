"""Automated downloader and extractor for UCSD Pedestrian (Ped2) Anomaly Dataset.

Downloads the official UCSD Anomaly Dataset tarball (~707 MB) from SVCL, extracts the UCSDped2
subset into `data/raw/UCSDped2`, and parses `UCSDped2.m` into a structured split manifest
at `data/annotations/ucsd_ped2_annotations.json`.
"""

import argparse
import json
import os
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vad.utils.logging import setup_logger

logger = setup_logger("vad.download_ucsd")

UCSD_URL = "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"


def download_file_with_resume(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with HTTP Range resume support and progress logging."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    initial_size = temp_path.stat().st_size if temp_path.exists() else 0

    headers = {"User-Agent": "Mozilla/5.0"}
    if initial_size > 0:
        headers["Range"] = f"bytes={initial_size}-"
        logger.info("Resuming download from byte %d (%.2f MB)...", initial_size, initial_size / (1024 * 1024))

    req = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range not satisfiable -> already completed
            logger.info("File already fully downloaded in temp path.")
            temp_path.replace(dest_path)
            return
        raise

    total_size = initial_size
    content_range = response.headers.get("Content-Range")
    content_length = response.headers.get("Content-Length")
    if content_range:
        total_size = int(content_range.split("/")[-1])
    elif content_length:
        total_size = initial_size + int(content_length)

    mode = "ab" if initial_size > 0 else "wb"
    downloaded = initial_size
    start_time = time.time()
    last_log_time = start_time

    with open(temp_path, mode) as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_log_time >= 5.0 or downloaded >= total_size:
                elapsed = max(now - start_time, 0.1)
                speed_mb = (downloaded - initial_size) / (1024 * 1024 * elapsed)
                pct = (downloaded / total_size * 100) if total_size > 0 else 0.0
                logger.info(
                    "Progress: %.1f%% (%.1f / %.1f MB) | Speed: %.2f MB/s",
                    pct,
                    downloaded / (1024 * 1024),
                    total_size / (1024 * 1024),
                    speed_mb,
                )
                last_log_time = now

    temp_path.replace(dest_path)
    logger.info("Download completed successfully: %s", dest_path)


def extract_ped2_from_tar(tar_path: Path, extract_dir: Path) -> Path:
    """Extract only the UCSDped2 portion from UCSD_Anomaly_Dataset.tar.gz."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    target_folder = extract_dir / "UCSDped2"

    logger.info("Opening archive %s to extract UCSDped2...", tar_path)
    extracted_count = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        members_to_extract = []
        for member in tar:
            # We look for files inside UCSD_Anomaly_Dataset.v1p2/UCSDped2/
            parts = Path(member.name).parts
            if "UCSDped2" in parts:
                ped2_idx = parts.index("UCSDped2")
                rel_parts = parts[ped2_idx + 1 :]
                if rel_parts:
                    member.name = str(Path(*rel_parts))
                    members_to_extract.append(member)

        logger.info("Found %d items in UCSDped2 to extract...", len(members_to_extract))
        for m in members_to_extract:
            tar.extract(m, path=target_folder)
            extracted_count += 1
            if extracted_count % 1000 == 0:
                logger.info("Extracted %d / %d files...", extracted_count, len(members_to_extract))

    logger.info("UCSDped2 extracted successfully to %s (%d files).", target_folder, extracted_count)
    return target_folder


def build_ucsd_ped2_split_manifest(ped2_dir: Path, output_json: Path) -> Path:
    """Parse UCSDped2.m and build train/val/test split annotations JSON."""
    m_file = ped2_dir / "Test" / "UCSDped2.m"
    if not m_file.exists():
        # Search anywhere in ped2_dir
        candidates = list(ped2_dir.rglob("*.m"))
        if candidates:
            m_file = candidates[0]
        else:
            raise FileNotFoundError(f"UCSDped2.m not found in {ped2_dir}")

    content = m_file.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"gt_frame\s*=\s*\[([^\]]*)\]", content)

    test_annotations: Dict[str, dict] = {}
    for idx, match_str in enumerate(matches, start=1):
        vid_id = f"Test{idx:03d}"
        intervals: List[List[int]] = []
        for s_str, e_str in re.findall(r"(\d+)\s*:\s*(\d+)", match_str):
            s_zero = max(0, int(s_str) - 1)
            e_zero = max(s_zero, int(e_str) - 1)
            intervals.append([s_zero, e_zero])

        test_annotations[vid_id] = {
            "num_frames": 180,
            "anomaly_intervals": intervals,
            "video_label": 1 if intervals else 0,
            "category": "anomalous" if intervals else "normal",
        }

    manifest = {"dataset": "UCSDped2", "videos": {}}

    # 1. Normal Train sequences: Train001 - Train016
    train_dirs = sorted([d for d in (ped2_dir / "Train").iterdir() if d.is_dir()])
    for idx, t_dir in enumerate(train_dirs, start=1):
        vid_id = t_dir.name
        # Assign 12 train normal, 2 val normal, 2 test normal
        if idx <= 12:
            split = "train"
        elif idx <= 14:
            split = "val"
        else:
            split = "test"

        manifest["videos"][vid_id] = {
            "split": split,
            "video_label": 0,
            "num_frames": 180,
            "anomaly_intervals": [],
            "category": "normal_pedestrian",
        }

    # 2. Test sequences with anomalies: Test001 - Test012
    # To enable supervised/weakly-supervised discriminative fusion training:
    # Test001 - Test006: train (6 anomalous sequences)
    # Test007 - Test008: val (2 anomalous sequences)
    # Test009 - Test012: test (4 anomalous sequences for final evaluation)
    test_dirs = sorted([d for d in (ped2_dir / "Test").iterdir() if d.is_dir() and not d.name.endswith("_gt")])
    for idx, t_dir in enumerate(test_dirs, start=1):
        vid_id = t_dir.name
        if idx <= 6:
            split = "train"
        elif idx <= 8:
            split = "val"
        else:
            split = "test"

        info = test_annotations.get(vid_id, {"anomaly_intervals": [], "video_label": 1, "category": "anomalous"})
        manifest["videos"][vid_id] = {
            "split": split,
            "video_label": info.get("video_label", 1),
            "num_frames": 180,
            "anomaly_intervals": info.get("anomaly_intervals", []),
            "category": info.get("category", "anomalous"),
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved UCSDped2 split manifest with %d videos to %s", len(manifest["videos"]), output_json)
    return output_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and prepare UCSD Ped2 anomaly dataset.")
    parser.add_argument("--tar-path", type=str, default="data/raw/UCSD_Anomaly_Dataset.tar.gz")
    parser.add_argument("--extract-dir", type=str, default="data/raw")
    parser.add_argument("--annotations-file", type=str, default="data/annotations/ucsd_ped2_annotations.json")
    parser.add_argument("--skip-download-if-exists", action="store_true", default=True)
    args = parser.parse_args()

    tar_path = Path(args.tar_path)
    extract_dir = Path(args.extract_dir)
    ped2_dir = extract_dir / "UCSDped2"
    ann_file = Path(args.annotations_file)

    if not (ped2_dir / "Train").exists() or not (ped2_dir / "Test").exists():
        if not tar_path.exists() or tar_path.stat().st_size < 700 * 1024 * 1024:
            logger.info("Downloading %s ...", UCSD_URL)
            download_file_with_resume(UCSD_URL, tar_path)
        else:
            logger.info("Found existing tarball at %s (%.1f MB)", tar_path, tar_path.stat().st_size / (1024 * 1024))

        logger.info("Extracting UCSDped2 from %s ...", tar_path)
        extract_ped2_from_tar(tar_path, extract_dir)
    else:
        logger.info("Found existing UCSDped2 directory at %s", ped2_dir)

    logger.info("Generating annotation splits manifest...")
    build_ucsd_ped2_split_manifest(ped2_dir, ann_file)
    logger.info("UCSD Ped2 dataset setup is complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
