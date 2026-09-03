# REPRODUCIBILITY & PROVENANCE REPORT

**Project:** HEDO-HVSC (Hamiltonian-Inspired Energy Dissipation and Chunk-Wise Variational State Coupling)  
**Target Venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence / Transactions on Multimedia  
**Dataset:** Official Flickr8k Split (6,000 Train, 1,000 Validation, 1,000 Held-Out Test, 5 Captions/Image)  
**Certified Leakage Status:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ (Zero Leakage)  
**Execution Environment:** Python 3.12.13, PyTorch 2.10.0+cu128, timm 1.0.26, transformers 5.0.0  
**Hardware:** NVIDIA Tesla T4 GPU (16 GB VRAM)  

---

## 1. Full Hyperparameter Configuration

| Parameter | Value |
| :--- | :--- |
| Visual Backbone | Pretrained `vit_base_patch16_224` (Frozen, 86.6 M params, $L_I=197$) |
| Textual Backbone | Pretrained `roberta-base` (Frozen, 124.6 M params, $L_T=64$) |
| Projection Dimension ($d_{\text{model}}$) | $128$ |
| Recurrence State Dimension ($d_{\text{state}}$) | $64$ |
| Chunk Size ($C$) | $16$ tokens |
| HEDO Time Step ($\Delta t$) | $0.1$ |
| HEDO Damping ($\beta$) | $0.05$ |
| HEDO Perturbation Scale ($\gamma$) | $0.1$ |
| Latent Dimension ($z_{\text{dim}}$) | $64$ |
| Symmetric KL Weight ($\lambda_{\text{KL}}$) | $0.01$ (strictly computed in FP32) |
| Optimizer | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-4}$) |
| Batch Size | $64$ images ($320$ captions/batch via 5:1 multi-positive InfoNCE) |
| Training Schedule | Cosine Annealing ($\eta_{\text{max}}=2\times 10^{-4}, \eta_{\text{min}}=10^{-6}$, 8 epochs) |
| Precision | Automatic Mixed Precision (AMP FP16) with GradScaler |
| Early Stopping | Full Validation Mean Recall (Patience: 3 epochs, $\Delta_{\text{min}}=0.05\%$) |
| Multi-Seed Protocol | 3 independent random seeds ($42, 43, 44$) across 4 configurations ($12$ total runs) |

---

## 2. Complete 12-Run Benchmark Results (`master_results.csv`)

| Configuration | Seed | Best Epoch | Full Val MR | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | Test MR | Train Time | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SSD Baseline** | 42 | 8 | 48.03% | 27.00% | 57.60% | 70.10% | 21.24% | 50.16% | 65.18% | 48.55% | 17.51 min | 1875.76 MB |
| | 43 | 8 | 48.01% | 26.30% | 56.20% | 68.60% | 19.94% | 49.78% | 64.68% | 47.58% | 17.47 min | 1878.22 MB |
| | 44 | 8 | 47.78% | 26.30% | 58.10% | 71.60% | 20.18% | 49.74% | 65.74% | 48.61% | 17.47 min | 1877.78 MB |
| **HEDO-HVSC w/o HEDO** | 42 | 8 | 48.08% | 27.70% | 57.90% | 69.30% | 20.52% | 50.14% | 65.02% | 48.43% | 17.72 min | 1878.61 MB |
| | 43 | 8 | 48.14% | 25.70% | 55.90% | 69.20% | 19.78% | 50.04% | 65.08% | 47.62% | 17.84 min | 1878.37 MB |
| | 44 | 8 | 47.67% | 26.70% | 57.40% | 71.60% | 20.58% | 49.66% | 65.72% | 48.61% | 17.90 min | 1878.81 MB |
| **HEDO-HVSC w/o HVSC** | 42 | 8 | 48.03% | 27.90% | 56.70% | 69.80% | 21.28% | 50.08% | 64.84% | 48.43% | 17.56 min | 1878.99 MB |
| | 43 | 8 | 49.16% | 25.80% | 57.10% | 70.20% | 19.98% | 50.28% | 65.04% | 48.07% | 17.56 min | 1879.74 MB |
| | 44 | 8 | 48.70% | 27.90% | 57.10% | 70.90% | 20.72% | 50.62% | 65.42% | 48.78% | 17.60 min | 2685.66 MB |
| **Full HEDO-HVSC (Ours)** | 42 | 8 | 47.86% | 28.20% | 56.50% | 69.90% | 20.88% | 50.14% | 64.84% | 48.41% | 17.81 min | 2686.05 MB |
| | 43 | 7 | 48.56% | 24.10% | 56.50% | 69.90% | 20.46% | 49.42% | 64.62% | 47.50% | 17.80 min | 2686.25 MB |
| | 44 | 8 | 48.60% | 27.90% | 57.50% | 71.60% | 20.78% | 50.36% | 65.42% | 48.93% | 17.85 min | 2686.25 MB |
