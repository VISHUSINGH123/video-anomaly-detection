# Troubleshooting Guide

## 1. Device & GPU Issues (AMD / Intel Laptops vs. NVIDIA CUDA)

### Symptom
`Requested device 'cuda' but CUDA is not available on this machine. Falling back to CPU execution.`

### Cause & Resolution
PyTorch's standard `cuda` backend requires an NVIDIA GPU with CUDA drivers. On laptops equipped with an **AMD Ryzen / Radeon GPU** or Intel integrated graphics:
- Use `configs/cpu.yaml` or `--device auto` (which automatically detects available hardware and falls back cleanly to CPU without crashing).
- Optional Windows AMD GPU acceleration: Install `torch-directml` (`pip install torch-directml`) and pass `--device directml` or `--device auto`.

---

## 2. Hugging Face Model Download & Offline Execution

### Symptom
`RuntimeError: Failed to load VideoMAE checkpoint 'MCG-NJU/videomae-base'. Ensure internet connectivity for initial download (~345 MB) or cache the checkpoint locally.`

### Cause & Resolution
1. **First-time download**: Ensure your machine has internet access to `huggingface.co` and ~700 MB of free disk space for `MCG-NJU/videomae-base` (~345 MB) and `facebook/dinov2-base` (~346 MB).
2. **Air-gapped / Offline machines**: Download both repositories on a connected machine using `huggingface-cli download MCG-NJU/videomae-base` and `huggingface-cli download facebook/dinov2-base`, copy the cache folder to `.cache/huggingface`, and set `local_files_only: true` in `configs/default.yaml`.
3. **Lightweight Smoke Testing**: Pass `--synthetic-encoders` to `scripts/extract_features.py` or `scripts/predict_video.py`, or enable **"Synthetic Smoke-Test Mode"** in the Streamlit sidebar to test the pipeline without downloading pretrained checkpoints.

---

## 3. Out-of-Memory (OOM) During Feature Extraction

### Symptom
`RuntimeError: CUDA out of memory` or high system RAM usage on long videos.

### Resolution
- Reduce `encoders.videomae.batch_size` (e.g. from `4` to `1` or `2`) and `encoders.dinov2.batch_size` (e.g. from `16` to `4`) in your YAML config.
- Note that `VideoMAEEncoder.extract_clips()` includes automatic OOM recovery that clears the CUDA cache and retries clip-by-clip (`batch_size=1`) when a batch OOM occurs.

---

## 4. Video Decoding & Corrupt File Errors

### Symptom
`RuntimeError: Failed to open or decode video file (corrupt or unsupported)` or `Video file contained zero readable frames`.

### Resolution
- Verify that the video file is non-empty and encoded with a standard codec supported by OpenCV/FFmpeg (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) or is a directory of sequential frame images (`.tif`, `.png`, `.jpg`).
- If a video has fewer frames than `clip_length` (16), ensure `data.short_video_policy: "pad"` is set in your YAML configuration so the last frame is repeated to form a valid 16-frame window.

---

## 5. Single-Class Evaluation Warning

### Symptom
`Frame ROC-AUC : N/A` and `Notes: Frame ROC-AUC undefined (ground truth contains only a single class).`

### Cause & Resolution
ROC-AUC requires at least one normal frame (`0`) and one anomalous frame (`1`) in the evaluated split. If you evaluate exclusively on normal videos (or without ground-truth anomaly annotations loaded), `roc_auc_score` is mathematically undefined. Ensure `--annotations-dir` points to valid ground-truth annotations containing both classes.
