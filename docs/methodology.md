# Methodology: Mathematical Formulation of Multi-Prior Temporal Anomaly Detection

## 1. Problem Formulation

Let a surveillance video $\mathcal{V} = \{I_0, I_1, \dots, I_{N-1}\}$ consist of $N$ RGB frames recorded at frame rate $F_{\text{fps}}$. We partition $\mathcal{V}$ into $T$ overlapping temporal windows (clips) $\{\mathcal{C}_t\}_{t=1}^{T}$ of fixed length $L$ frames (default $L = 16$) and stride $S$ frames (default $S = 8$). Each window $\mathcal{C}_t$ spans frame indices $[a_t, b_t]$ and timestamps $[\tau_t^{\text{start}}, \tau_t^{\text{end}}]$.

The goal is to learn a mapping $\Phi(\mathcal{V}) \mapsto \{\hat{s}_t\}_{t=1}^{T} \in (0, 1)^T$ that assigns a continuous anomaly probability $\hat{s}_t$ to each temporal window $\mathcal{C}_t$, and subsequently projects $\{\hat{s}_t\}$ onto frame-level anomaly scores $\{\hat{y}_i\}_{i=0}^{N-1} \in [0, 1]^N$.

---

## 2. Dual Foundation Feature Extraction & Alignment

### 2.1 Spatiotemporal Semantic Stream (VideoMAE)
For each clip $\mathcal{C}_t$, all $L = 16$ frames are resized to $224 \times 224$ and normalized using ImageNet channel statistics. The frozen pretrained VideoMAE encoder $\mathcal{E}_{\text{v}}$ (`MCG-NJU/videomae-base`) partitions the $16 \times 3 \times 224 \times 224$ volume into $M = \frac{16}{2} \times \frac{224}{16} \times \frac{224}{16} = 1568$ spatiotemporal tubelet tokens $\mathbf{U}_t \in \mathbb{R}^{M \times D_v}$ ($D_v = 768$). We extract the mean-pooled spatiotemporal descriptor:

$$
\mathbf{f}_t^{(v)} = \frac{1}{M} \sum_{m=1}^{M} \mathbf{U}_{t, m} \in \mathbb{R}^{D_v}
$$

### 2.2 Frame-Level Appearance Stream (DINOv2)
Within the same clip $\mathcal{C}_t$, we sample $K = \lceil L / r \rceil$ frames at stride $r$ (`frame_sample_rate`) and pass each frame through the frozen pretrained DINOv2 ViT-B/14 encoder $\mathcal{E}_{\text{d}}$ (`facebook/dinov2-base`), extracting the global CLS token $\mathbf{c}_{t, k} \in \mathbb{R}^{D_d}$ ($D_d = 768$) for $k \in \{1, \dots, K\}$. To preserve both static scene appearance and rapid intra-clip visual variation, we compute:

$$
\boldsymbol{\mu}_t^{(d)} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{c}_{t, k}, \qquad \boldsymbol{\sigma}_t^{(d)} = \sqrt{\frac{1}{K} \sum_{k=1}^{K} \left(\mathbf{c}_{t, k} - \boldsymbol{\mu}_t^{(d)}\right)^2}
$$

$$
\mathbf{f}_t^{(d)} = \begin{cases}
\boldsymbol{\mu}_t^{(d)} + \frac{1}{2} \boldsymbol{\sigma}_t^{(d)} & \text{if pooling = \texttt{cls\_and\_std}} \\
\boldsymbol{\mu}_t^{(d)} & \text{if pooling = \texttt{cls}}
\end{cases} \in \mathbb{R}^{D_d}
$$

Because both $\mathbf{f}_t^{(v)}$ and $\mathbf{f}_t^{(d)}$ are computed over the exact same window $\mathcal{C}_t$, the two feature streams are strictly aligned in time at index $t \in \{1, \dots, T\}$.

---

## 3. Stream Normalization & Projection

Foundation model embeddings from distinct pretraining paradigms exhibit different $L_2$ norms and channel variances. Before fusion, each stream is $L_2$-normalized and projected into a shared latent space of dimension $D_p = 256$:

$$
\mathbf{h}_t^{(v)} = \text{Dropout}\!\left(\text{GELU}\!\left(\text{LayerNorm}\!\left(\mathbf{W}_v \frac{\mathbf{f}_t^{(v)}}{\|\mathbf{f}_t^{(v)}\|_2 + \epsilon} + \mathbf{b}_v\right)\right)\right) \in \mathbb{R}^{D_p}
$$

$$
\mathbf{h}_t^{(d)} = \text{Dropout}\!\left(\text{GELU}\!\left(\text{LayerNorm}\!\left(\mathbf{W}_d \frac{\mathbf{f}_t^{(d)}}{\|\mathbf{f}_t^{(d)}\|_2 + \epsilon} + \mathbf{b}_d\right)\right)\right) \in \mathbb{R}^{D_p}
$$

---

## 4. Multi-Scale Dilated Temporal Convolutional Network (TCN)

First, the projected spatiotemporal and appearance streams are combined into a joint clip embedding $\mathbf{u}_t = \text{GELU}(\text{LayerNorm}(\mathbf{W}_{\text{in}} [\mathbf{h}_t^{(v)} \,\|\, \mathbf{h}_t^{(d)}] + \mathbf{b}_{\text{in}})) \in \mathbb{R}^{D_p}$.

The sequence $\mathbf{U} = [\mathbf{u}_1, \dots, \mathbf{u}_T]^\top \in \mathbb{R}^{T \times D_p}$ is processed by a stack of $L_{\text{tcn}}$ residual 1D dilated convolutional blocks with kernel size $K_{\text{tcn}}$ (default $3$) and exponential dilation schedule $d_\ell = 2^{\ell}$ for $\ell \in \{0, \dots, L_{\text{tcn}}-1\}$:

$$
\mathbf{Z}^{(\ell+1)} = \mathcal{B}_\ell(\mathbf{Z}^{(\ell)}) + \mathcal{R}_\ell(\mathbf{Z}^{(\ell)})
$$

where $\mathcal{B}_\ell$ applies two dilated 1D convolutions with `GroupNorm`, `GELU`, and dropout, and $\mathcal{R}_\ell$ is the identity or $1\times 1$ channel projection. The resulting temporal context prior is:

$$
\mathbf{h}_t^{(t)} = \mathbf{W}_{\text{tcn\_out}}\, \text{LayerNorm}(\mathbf{Z}^{(L_{\text{tcn}})}_t) \in \mathbb{R}^{D_p}
$$

The effective temporal receptive field in clips is:

$$
R_{\text{clips}} = 1 + 2 (K_{\text{tcn}} - 1) \sum_{\ell=0}^{L_{\text{tcn}}-1} 2^\ell
$$

For $L_{\text{tcn}} = 4$ and $K_{\text{tcn}} = 3$, $R_{\text{clips}} = 1 + 4(1 + 2 + 4 + 8) = 61$ clips.

---

## 5. Multi-Prior Gated Feature Fusion

In addition to $\mathbf{h}_t^{(v)}$ (spatiotemporal semantic prior), $\mathbf{h}_t^{(d)}$ (appearance prior), and $\mathbf{h}_t^{(t)}$ (long-range TCN temporal context prior), we compute a fourth explicit **temporal feature-change (motion/novelty velocity) prior** $\mathbf{h}_t^{(m)}$ from first-order temporal differences:

$$
\Delta \mathbf{h}_t^{(v)} = \mathbf{h}_t^{(v)} - \mathbf{h}_{t-1}^{(v)}, \qquad \Delta \mathbf{h}_t^{(d)} = \mathbf{h}_t^{(d)} - \mathbf{h}_{t-1}^{(d)} \quad (\text{with } \Delta \mathbf{h}_1 = \mathbf{0})
$$

$$
\mathbf{h}_t^{(m)} = \text{Dropout}\!\left(\text{GELU}\!\left(\text{LayerNorm}\!\left(\mathbf{W}_m [\Delta \mathbf{h}_t^{(v)} \,\|\, \Delta \mathbf{h}_t^{(d)}] + \mathbf{b}_m\right)\right)\right) \in \mathbb{R}^{D_p}
$$

Let the four aligned prior embeddings at clip $t$ be $\mathcal{P}_t = \{\mathbf{h}_t^{(v)}, \mathbf{h}_t^{(d)}, \mathbf{h}_t^{(t)}, \mathbf{h}_t^{(m)}\}$ and let $\mathbf{c}_t = [\mathbf{h}_t^{(v)} \,\|\, \mathbf{h}_t^{(d)} \,\|\, \mathbf{h}_t^{(t)} \,\|\, \mathbf{h}_t^{(m)}] \in \mathbb{R}^{4 D_p}$. A two-layer gating network predicts temperature-scaled softmax gate weights $\boldsymbol{\alpha}_t = [\alpha_{t,1}, \alpha_{t,2}, \alpha_{t,3}, \alpha_{t,4}]^\top \in \Delta^3$:

$$
\mathbf{g}_t = \frac{1}{\tau} \left(\mathbf{W}_{g2}\, \text{GELU}(\text{LayerNorm}(\mathbf{W}_{g1} \mathbf{c}_t + \mathbf{b}_{g1})) + \mathbf{b}_{g2}\right) \in \mathbb{R}^4
$$

$$
\alpha_{t, k} = \frac{m_k \exp(g_{t, k})}{\sum_{j=1}^{4} m_j \exp(g_{t, j})}, \qquad \sum_{k=1}^{4} \alpha_{t, k} = 1
$$

where $m_k \in \{0, 1\}$ is the binary ablation mask enabling or disabling specific streams. The fused representation $\mathbf{z}_t \in \mathbb{R}^{D_p}$ and continuous clip anomaly score $\hat{s}_t \in (0, 1)$ are:

$$
\mathbf{z}_t = \text{LayerNorm}\!\left(\sum_{k=1}^{4} \alpha_{t, k} \mathbf{h}_t^{(k)}\right), \qquad \hat{s}_t = \sigma(\ell_t) = \sigma\!\left(\mathbf{w}_s^\top \text{GELU}(\text{LayerNorm}(\mathbf{W}_{s1} \mathbf{z}_t + \mathbf{b}_{s1})) + b_s\right)
$$

---

## 6. Supervision & Learning Objectives

### 6.1 Supervised Clip-Level Learning (`supervision: supervised_clip`)
When frame-level ground-truth intervals are available for training/validation splits, each clip $t$ receives a binary target $y_t \in \{0, 1\}$ (1 if at least $25\%$ of frames in $\mathcal{C}_t$ are anomalous). To address positive-class scarcity, we optimize a positive-weighted Binary Cross-Entropy loss combined with temporal smoothness and sparsity regularization:

$$
\mathcal{L}_{\text{clip}} = \frac{1}{|\mathcal{V}_{\text{valid}}|} \sum_{t \in \mathcal{V}_{\text{valid}}} \left[ -w_{\text{pos}} y_t \log \hat{s}_t - (1 - y_t) \log(1 - \hat{s}_t) \right]
$$

### 6.2 Weakly-Supervised Top-$K$ Multiple-Instance Learning (`supervision: weakly_supervised_mil`)
When only video-level binary labels $Y_i \in \{0, 1\}$ are available (e.g., standard UCF-Crime / ShanghaiTech weak supervision), each video $i$ of length $T_i$ is treated as a bag of clips. We compute the mean of the top $k_i = \lceil \rho T_i \rceil$ clip logits ($\rho = 0.15$):

$$
\bar{\ell}_i = \frac{1}{k_i} \sum_{t \in \text{TopK}(\{\ell_{i,t}\}, k_i)} \ell_{i, t}, \qquad \mathcal{L}_{\text{MIL}} = \frac{1}{B} \sum_{i=1}^{B} \text{BCEWithLogits}(\bar{\ell}_i, Y_i; w_{\text{pos}})
$$

### 6.3 Temporal Smoothness & Sparsity Regularization
Because anomalous events are contiguous in time and rare overall, both supervision modes include temporal smoothness ($\lambda_{\text{smooth}} = 0.05$) and $L_1$ sparsity ($\lambda_{\text{sparse}} = 0.005$) penalties:

$$
\mathcal{L}_{\text{smooth}} = \frac{1}{T - 1} \sum_{t=2}^{T} (\hat{s}_t - \hat{s}_{t-1})^2, \qquad \mathcal{L}_{\text{sparse}} = \frac{1}{T} \sum_{t=1}^{T} \hat{s}_t
$$

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls/MIL}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{sparse}} \mathcal{L}_{\text{sparse}}
$$

---

## 7. Frame-Level Projection

Given smoothed clip scores $\{\tilde{s}_t\}_{t=1}^{T}$ and window index sets $\{\mathcal{I}_t\}_{t=1}^{T}$, the continuous anomaly score for frame $i \in \{0, \dots, N-1\}$ is the average over all covering windows:

$$
\hat{y}_i = \frac{\sum_{t : i \in \mathcal{I}_t} \tilde{s}_t}{\left|\{t : i \in \mathcal{I}_t\}\right|}
$$
