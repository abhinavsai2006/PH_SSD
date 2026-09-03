# REPRODUCIBILITY GUIDE & PROVENANCE REPORT

**Project:** HEDO-HVSC (Hamiltonian-Inspired Energy Dissipation & Chunk-Wise Variational State Coupling)  
**Dataset:** Official Flickr8k (Zero Split Leakage: 6,000 Train, 1,000 Validation, 1,000 Held-Out Test)  
**Hardware:** NVIDIA Tesla T4 GPU (16 GB VRAM)  
**Software Environment:** Python 3.12.13, PyTorch 2.10.0+cu128, timm 1.0.26, transformers 5.0.0  

---

## 1. Experimental Configuration & Hyperparameters

| Hyperparameter | Value / Specification |
| :--- | :--- |
| **Visual Encoder** | Frozen `timm` Vision Transformer (`vit_base_patch16_224`, 86.6 M parameters, $L_I=197$) |
| **Textual Encoder** | Frozen HuggingFace RoBERTa (`roBERTa-base`, 124.6 M parameters, $L_T=64$) |
| **Model Embedding Dimension ($d_{\text{model}}$)** | $128$ |
| **State Dimension ($d_{\text{state}}$)** | $64$ |
| **Chunk Size ($C$)** | $16$ tokens |
| **HEDO Integration Step ($\Delta t$)** | $0.1$ |
| **HEDO Damping Coefficient ($\beta$)** | $0.05$ |
| **HEDO Perturbation Scale ($\gamma$)** | $0.1$ |
| **Latent Space Dimension ($z_{\text{dim}}$)** | $64$ |
| **KL Divergence Weight ($\lambda_{\text{KL}}$)** | $0.01$ (FP32 precision) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-4}$) |
| **Batch Size** | $64$ ($320$ captions/batch via 5:1 multi-positive pairing) |
| **Learning Rate Schedule** | Cosine Annealing ($\eta_{\text{max}}=2\times 10^{-4}, \eta_{\text{min}}=10^{-6}$, 8 epochs) |
| **Precision** | AMP FP16 with GradScaler (KL divergence strictly in FP32) |
| **Gradient Clipping** | Maximum norm $1.0$ |
| **Early Stopping** | Monitored on full validation Mean Recall (patience: 3 epochs, $\Delta_{\text{min}}=0.05\%$) |

---

## 2. Certified 12-Run Benchmark Results (`master_results.csv`)

| Model Configuration | Seed | Best Epoch | Full Val MR (%) | Test I2T R@1 (%) | Test T2I R@1 (%) | Test MR (%) | Train Time | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SSD Baseline** | 42 | 8 | 48.03 | 27.00 | 21.24 | 48.55 | 17.51 min | 1875.76 MB |
| | 43 | 8 | 48.01 | 26.30 | 19.94 | 47.58 | 17.47 min | 1878.22 MB |
| | 44 | 8 | 47.78 | 26.30 | 20.18 | 48.61 | 17.47 min | 1877.78 MB |
| **HEDO-HVSC w/o HEDO** | 42 | 8 | 48.08 | 27.70 | 20.52 | 48.43 | 17.72 min | 1878.61 MB |
| | 43 | 8 | 48.14 | 25.70 | 19.78 | 47.62 | 17.84 min | 1878.37 MB |
| | 44 | 8 | 47.67 | 26.70 | 20.58 | 48.61 | 17.90 min | 1878.81 MB |
| **HEDO-HVSC w/o HVSC** | 42 | 8 | 48.03 | 27.90 | 21.28 | 48.43 | 17.56 min | 1878.99 MB |
| | 43 | 8 | 49.16 | 25.80 | 19.98 | 48.07 | 17.56 min | 1879.74 MB |
| | 44 | 8 | 48.70 | 27.90 | 20.72 | 48.78 | 17.60 min | 2685.66 MB |
| **Full HEDO-HVSC (Ours)** | 42 | 8 | 47.86 | 28.20 | 20.88 | 48.41 | 17.81 min | 2686.05 MB |
| | 43 | 7 | 48.56 | 24.10 | 20.46 | 47.50 | 17.80 min | 2686.25 MB |
| | 44 | 8 | 48.60 | 27.90 | 20.78 | 48.93 | 17.85 min | 2686.25 MB |

---

## 3. Reproduction Instructions

1. Ensure Python 3.12+ and PyTorch with CUDA support are installed.
2. Compile LaTeX manuscript and supplementary materials:
   ```bash
   pdflatex -interaction=nonstopmode main.tex
   bibtex main
   pdflatex -interaction=nonstopmode main.tex
   pdflatex -interaction=nonstopmode main.tex
   pdflatex -interaction=nonstopmode supplementary_material.tex
   ```
