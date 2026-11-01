# Research-Grade Experiments & Ablation Study Protocol

## 1. Experimental Variants (Ablation Design)

To rigorously isolate the contribution of each architectural component, all five model variants must be trained and evaluated on the **exact same dataset split, clip length ($L=16$), clip stride ($S=8$), random seed (`seed=42`), and evaluation protocol**:

| Variant ID | Model Name | CLI `--fusion-mode` / Config | Active Streams |
| :--- | :--- | :--- | :--- |
| **A** | **VideoMAE-only Baseline** | `--fusion-mode videomae_only` | VideoMAE (`768 -> 256`) + Scoring Head (`alpha = [1, 0, 0, 0]`) |
| **B** | **DINOv2-only Baseline** | `--fusion-mode dinov2_only` | DINOv2 (`768 -> 256`) + Scoring Head (`alpha = [0, 1, 0, 0]`) |
| **C** | **Single-Stream + TCN** | `encoders.dinov2.enabled=false`, `--fusion-mode gated_multiprior` | VideoMAE + Dilated TCN (`L=4, K=3`) + Scoring Head |
| **D** | **Dual-Stream Fusion without TCN** | `--fusion-mode no_tcn` | VideoMAE + DINOv2 + Motion Velocity (TCN disabled, `alpha_t = 0`) |
| **E** | **Full Multi-Prior Fusion Model** | `--fusion-mode gated_multiprior` | VideoMAE + DINOv2 + Dilated TCN + Motion Velocity + Softmax Gate |

---

## 2. Automated Ablation Runner & Individual Commands

You can run all 5 variants automatically with a single command:
```bash
python scripts/run_ablations.py --config configs/cpu.yaml --epochs 8
```
This trains all variants, benchmarks per-clip latency, saves checkpoints to `checkpoints/<variant_id>/`, copies the top model to `checkpoints/best.pt`, exports `outputs/evaluation/ablations/ablation_summary.csv`, and generates the comparative chart `outputs/evaluation/ablations/ablation_comparison.png`.

Alternatively, run each variant individually once features have been cached in `data/processed/` via `scripts/extract_features.py`:

```bash
# Variant A: VideoMAE-only Baseline
python scripts/train.py --config configs/default.yaml --fusion-mode videomae_only --checkpoint-dir checkpoints/exp_A_videomae
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/exp_A_videomae/best.pt --output-dir outputs/evaluation/exp_A_videomae

# Variant B: DINOv2-only Baseline
python scripts/train.py --config configs/default.yaml --fusion-mode dinov2_only --checkpoint-dir checkpoints/exp_B_dinov2
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/exp_B_dinov2/best.pt --output-dir outputs/evaluation/exp_B_dinov2

# Variant C: Concat-MLP (No Gating)
python scripts/train.py --config configs/default.yaml --fusion-mode concat_mlp --checkpoint-dir checkpoints/exp_C_concat_mlp
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/exp_C_concat_mlp/best.pt --output-dir outputs/evaluation/exp_C_concat_mlp

# Variant D: VideoMAE + DINOv2 Fusion without TCN
python scripts/train.py --config configs/default.yaml --fusion-mode no_tcn --checkpoint-dir checkpoints/exp_D_no_tcn
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/exp_D_no_tcn/best.pt --output-dir outputs/evaluation/exp_D_no_tcn

# Variant E: Full VideoMAE + DINOv2 + TCN + Multi-Prior Gated Fusion
python scripts/train.py --config configs/default.yaml --fusion-mode gated_multiprior --checkpoint-dir checkpoints/exp_E_full
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/exp_E_full/best.pt --output-dir outputs/evaluation/exp_E_full
```

---

## 3. Benchmark Results on UCSD Pedestrian 2 (Ped2)

The benchmark evaluation below was performed on the official **UCSD Pedestrian 2 (Ped2)** dataset using frozen **VideoMAE** (`MCG-NJU/videomae-base`) and **DINOv2** (`facebook/dinov2-base`) feature representations:

| Variant | Architecture / Mode | Frame ROC-AUC (%) | Frame PR-AUC (%) | Frame F1 (@0.50) | Video ROC-AUC (%) | Train Time (s) | Inference Latency (ms/clip) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. VideoMAE-only** | `videomae_only` | 88.34% | 90.54% | 0.00% | 100.0% | 1.39s | 0.16 ms |
| **B. DINOv2-only** | `dinov2_only` | 99.24% | 99.43% | 63.74% | 100.0% | 1.22s | 0.19 ms |
| **C. Concat-MLP** | `concat_mlp` | **99.76%** | **99.81%** | 66.42% | 100.0% | 3.90s | 0.61 ms |
| **D. Dual-Stream (No TCN)** | `no_tcn` | 99.24% | 99.43% | 63.74% | 100.0% | 1.29s | 0.14 ms |
| **E. Full Multi-Prior Fusion** | `gated_multiprior` | 99.24% | 99.43% | **76.17%** | 100.0% | 3.47s | 0.55 ms |

* Full evaluation artifacts and per-video metrics: `outputs/evaluation/ucsd_ped2/`
* Comparative visualization plot: `outputs/evaluation/ucsd_ped2/ablation_comparison.png`

---

## 4. Inspecting Learned Gate Weights

Every evaluation run (`outputs/evaluation/<exp>/per_video_summary.csv`) and single-video inference run (`outputs/predictions/<video_id>_clip_scores.csv`) records the exact per-clip and mean prior gate weights:
- `gate_videomae` ($\alpha_v$)
- `gate_dinov2` ($\alpha_d$)
- `gate_tcn_temporal` ($\alpha_t$)
- `gate_motion_dynamics` ($\alpha_m$)

This allows researchers to inspect whether the model dynamically shifts weight toward motion/spatiotemporal priors during sudden fast anomalies versus appearance priors during static anomalies.
