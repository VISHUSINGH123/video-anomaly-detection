"""Environment audit script inspecting OS, Python, packages, disk space, and GPU/CPU acceleration."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

# Ensure `src` is importable when running directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vad.device import resolve_device


REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "scikit-learn",
    "torch",
    "torchvision",
    "transformers",
    "opencv-python-headless",
    "Pillow",
    "PyYAML",
    "pandas",
    "matplotlib",
    "tqdm",
    "streamlit",
    "pytest",
]


def check_packages() -> dict[str, str]:
    status: dict[str, str] = {}
    for pkg in REQUIRED_PACKAGES:
        try:
            status[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            # Handle opencv fallback if regular opencv-python is installed
            if pkg == "opencv-python-headless":
                try:
                    status[pkg] = importlib.metadata.version("opencv-python")
                    continue
                except importlib.metadata.PackageNotFoundError:
                    pass
            status[pkg] = "MISSING"
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit OS, Python version, installed packages, disk space, and compute devices."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device to probe (auto, cpu, cuda, directml, mps).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output environment report as formatted JSON.",
    )
    args = parser.parse_args()

    disk = shutil.disk_usage(PROJECT_ROOT)
    dev_info = resolve_device(args.device)
    pkg_status = check_packages()
    missing = [k for k, v in pkg_status.items() if v == "MISSING"]

    report = {
        "os_platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_processor": dev_info.cpu_processor,
        "workspace_path": str(PROJECT_ROOT),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "selected_device": str(dev_info.device),
        "device_type": dev_info.device_type,
        "device_name": dev_info.device_name,
        "cuda_available": dev_info.cuda_available,
        "directml_available": dev_info.directml_available,
        "mps_available": dev_info.mps_available,
        "supports_fp16": dev_info.supports_fp16,
        "packages": pkg_status,
        "missing_packages": missing,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 72)
        print("VIDEO ANOMALY DETECTION — ENVIRONMENT AUDIT")
        print("=" * 72)
        print(f"OS Platform       : {report['os_platform']}")
        print(f"Python Version    : {report['python_version']} ({report['python_executable']})")
        print(f"CPU Processor     : {report['cpu_processor']}")
        print(f"Disk Space (Free) : {report['disk_free_gb']} GB / {report['disk_total_gb']} GB")
        print("-" * 72)
        print(f"Resolved Device   : {report['selected_device']} ({report['device_name']})")
        print(f"CUDA Available    : {report['cuda_available']}")
        print(f"DirectML (AMD/Win): {report['directml_available']}")
        print(f"Apple MPS         : {report['mps_available']}")
        print("-" * 72)
        print("Installed Packages:")
        for pkg, ver in pkg_status.items():
            print(f"  - {pkg:<24}: {ver}")
        print("=" * 72)
        if missing:
            print(f"[ERROR] Missing required packages: {', '.join(missing)}")
            return 1
        print("[OK] Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
