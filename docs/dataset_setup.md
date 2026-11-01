# Dataset Setup & Preparation Guide

## 1. Supported Surveillance Datasets

This pipeline is designed to support established video anomaly detection benchmarks with varying annotation granularities:

| Dataset | Official Source | Annotation Granularity | Size | License / Usage |
| :--- | :--- | :--- | :--- | :--- |
| **UCSD Pedestrian (Ped1 / Ped2)** | [UCSD SVCL Anomaly Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) | Frame-level intervals (`UCSDped2.m`) + pixel masks | ~195 MB (`.tar.gz`) | Academic research & educational use |
| **ShanghaiTech Campus** | [SVIP-Lab ShanghaiTech Campus](https://svip-lab.github.io/dataset/campus_dataset.html) | Frame-level binary masks (`test_frame_mask/*.npy`) | ~13.5 GB | Non-commercial academic research |
| **UCF-Crime** | [UCF CRCV Real-World Anomaly Detection](https://www.crcv.ucf.edu/projects/real-world/) | Video-level labels (train) + temporal intervals (test `.txt`) | ~105 GB | Academic research use |

> **Important Note on Supervision Granularity:**
> In the original one-class UCSD Ped2 and ShanghaiTech splits, the `Train` folder contains only normal videos, while anomalous events appear in `Test`. To train our discriminative scoring head / multi-prior fusion module:
> 1. **Standard Weakly-Supervised / Supervised Split (ShanghaiTech-WS / UCF-Crime)**: Both normal and anomalous videos are present in the training split, allowing either `weakly_supervised_mil` (using video-level labels) or `supervised_clip` (when temporal intervals are provided).
> 2. **UCSD Ped2 Split for Discriminative Fusion**: Partition the annotated sequences (plus normal `Train` sequences) into disjoint `train`, `val`, and `test` subsets in `data/annotations/annotations.json` so that both training and evaluation splits contain normal and anomalous clips without video leakage.

---

## 2. UCSD Pedestrian (Ped2) Setup (Recommended Small-Scale Benchmark)

### 2.1 Download
Download `UCSD_Anomaly_Dataset.tar.gz` (~195 MB) from the official UCSD Statistical Visual Computing Lab page:
- URL: `http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz`

Extract the archive so that the Ped2 directories reside under `data/raw/`:
```text
data/raw/UCSDped2/
├── Train/
│   ├── Train001/ (001.tif ... 120.tif)
│   └── ...
└── Test/
    ├── Test001/ (001.tif ... 180.tif)
    ├── Test002/
    └── UCSDped2.m
```

### 2.2 Annotation Parsing
Our data pipeline (`src/vad/data/video_loader.py` and `src/vad/data/annotations.py`) natively:
- Discovers frame-sequence folders (`Train001`, `Test001`, etc.) containing `.tif` images and decodes them directly into RGB video volumes.
- Automatically skips `TestXXX_gt` pixel-mask directories so ground-truth masks are never mistaken for input videos.
- Parses the official MATLAB `UCSDped2.m` interval script (`TestVideoFile{end+1}.gt_frame = [61:180];`), converting 1-based MATLAB indices into 0-based inclusive frame intervals.

You can also provide a custom JSON split manifest at `data/annotations/ucsd_ped2_splits.json`:
```json
{
  "videos": {
    "Train001": {"split": "train", "video_label": 0, "anomaly_intervals": []},
    "Test001": {"split": "train", "video_label": 1, "anomaly_intervals": [[60, 179]]},
    "Test002": {"split": "val", "video_label": 1, "anomaly_intervals": [[94, 179]]},
    "Test003": {"split": "test", "video_label": 1, "anomaly_intervals": [[0, 145]]}
  }
}
```

---

## 3. ShanghaiTech Campus Dataset Setup

Place testing frame masks (`*.npy` where `0` = normal, `1` = anomalous) inside `data/annotations/shanghaitech/` and videos in `data/raw/shanghaitech/`. `load_annotations_directory()` automatically loads every `<video_id>.npy` file and maps it to a `VideoAnnotation` object.

---

## 4. Pretrained Foundation Model Checkpoints

During `scripts/extract_features.py` or `scripts/predict_video.py`, two pretrained Hugging Face models are required:

1. **VideoMAE (`MCG-NJU/videomae-base`)**
   - Source: `https://huggingface.co/MCG-NJU/videomae-base`
   - Download Size: ~345 MB
   - License: CC-BY-NC 4.0 (model weights) / Apache-2.0 (`transformers` integration)
2. **DINOv2 (`facebook/dinov2-base`)**
   - Source: `https://huggingface.co/facebook/dinov2-base`
   - Download Size: ~346 MB
   - License: Apache-2.0

To pre-cache both models when internet access is available:
```bash
python -c "from transformers import VideoMAEModel, AutoModel; VideoMAEModel.from_pretrained('MCG-NJU/videomae-base'); AutoModel.from_pretrained('facebook/dinov2-base')"
```

---

## 5. Synthetic Smoke-Test Dataset (Zero-Download Verification)

To verify the entire data -> feature caching -> TCN/fusion training -> evaluation -> video inference pipeline offline without downloading external datasets or 700 MB of foundation weights:

```bash
python scripts/download_sample_data.py --mode synthetic
```

All generated files are stored under `data/raw/synthetic/`, `data/processed/synthetic/`, and `data/annotations/synthetic/synthetic_annotations.json` with explicit `is_synthetic=True` flags.
