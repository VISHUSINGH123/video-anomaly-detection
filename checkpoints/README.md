# Model Checkpoints Directory

This directory stores trained PyTorch checkpoints (`.pt`) for the Temporal Convolutional Network (TCN) and Multi-Prior Fusion Anomaly Detector.

## Files Generated During Training
- `best.pt`: Checkpoint achieving the highest validation metric (Frame ROC-AUC or lowest validation loss).
- `last.pt`: Most recent epoch checkpoint for resuming interrupted runs.
- `training_history.json`: Epoch-by-epoch training/validation loss and metric log.

## Pretrained Foundation Backbones
The frozen feature extractors (`MCG-NJU/videomae-base` ~345 MB and `facebook/dinov2-base` ~346 MB) are loaded via Hugging Face `transformers` and cached in the standard Hugging Face cache (`HF_HOME` or `~/.cache/huggingface`).
