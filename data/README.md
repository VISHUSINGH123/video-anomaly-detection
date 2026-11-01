# Data Directory

This directory holds raw surveillance videos, extracted feature caches, and ground-truth annotations.
Large binary files are excluded from Git via `.gitignore`.

## Structure
- `data/raw/`: Place raw benchmark video files (`.mp4`, `.avi`) or frame directories (e.g., UCSD Pedestrian `Test001/*.tif`).
- `data/processed/`: Compressed `.npz` feature bundles produced by `scripts/extract_features.py`.
- `data/annotations/`: Ground-truth frame-level or video-level annotations (`.json`, `.csv`, UCSD `.m`, or ShanghaiTech `.npy` masks).

See [`docs/dataset_setup.md`](../docs/dataset_setup.md) for official dataset sources, licenses, and preparation steps.
