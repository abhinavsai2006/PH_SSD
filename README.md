# HEDO-HVSC: Hamiltonian-Inspired Energy Dissipation and Chunk-Wise Variational State Coupling for Efficient Multimodal State-Space Models

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)]()

This repository contains the official PyTorch implementation, configuration files, evaluation scripts, and reproducible benchmark manifests for the research paper:

> **Hamiltonian-Inspired Energy Dissipation and Chunk-Wise Variational State Coupling for Efficient Multimodal State-Space Models**  
> *Abhinav Sai Maddineni* (School of Computer Science and Engineering, VIT-AP University)

---

## 🔬 Core Architecture Components

1. **Hamiltonian-Inspired Energy Dissipation Operator (HEDO):**  
   A learned discrete coordinate-momentum sequence transformation with positive damping ($\beta=0.05, \Delta t=0.1, \gamma=0.1$) that introduces a dissipative inductive bias to reduce sensitivity to representation perturbations prior to sequence recurrence.

2. **Custom Chunk-Wise State-Space Duality (SSD) Recurrence:**  
   A pure PyTorch state-space sequence recurrence block ($C=16, d_{\text{state}}=64$) maintaining hidden-state continuity across chunk boundaries without inter-chunk resets. *(Note: The benchmark experiments use this pure PyTorch SSD-style recurrent implementation and do not require or evaluate the native `mamba_ssm` CUDA C extensions).*

3. **Chunk-Wise Variational State Coupling (HVSC):**  
   Mask-weighted pooling of chunk boundary states parameterizing Gaussian latent distributions $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ regularized via symmetric Kullback-Leibler (KL) divergence in FP32 precision during training, while enforcing strictly unimodal, deterministic inference at test time.

---

## 📊 Summary of Benchmark Results

All experiments are conducted across $N=3$ independent random seeds ($42, 43, 44$) on the predefined Flickr8k dataset partition ($6{,}000$ train, $1{,}000$ val, $1{,}000$ held-out test, 5 captions/image) with frozen ViT-B/16 and RoBERTa-base encoders:

| Model Configuration | HEDO | HVSC | I2T R@1 (%) | I2T R@5 (%) | I2T R@10 (%) | T2I R@1 (%) | T2I R@5 (%) | T2I R@10 (%) | Test Mean Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SSD Baseline** | No | No | $26.53 \pm 0.33$ | $57.30 \pm 0.80$ | $70.10 \pm 1.22$ | $20.45 \pm 0.56$ | $49.89 \pm 0.19$ | $65.20 \pm 0.43$ | **48.25 ± 0.47** |
| **HEDO-HVSC w/o HEDO** | No | Yes | $26.70 \pm 0.82$ | $57.07 \pm 0.85$ | $70.03 \pm 1.11$ | $20.29 \pm 0.36$ | $49.95 \pm 0.21$ | $65.27 \pm 0.32$ | **48.22 ± 0.43** |
| **HEDO-HVSC w/o HVSC** | Yes | No | $27.20 \pm 0.99$ | $56.97 \pm 0.19$ | $70.30 \pm 0.45$ | $20.66 \pm 0.53$ | $50.33 \pm 0.22$ | $65.10 \pm 0.24$ | **48.43 ± 0.29** |
| **Full HEDO-HVSC (Ours)** | Yes | Yes | $26.73 \pm 1.87$ | $56.83 \pm 0.47$ | $70.47 \pm 0.80$ | $20.71 \pm 0.18$ | $49.97 \pm 0.40$ | $64.96 \pm 0.34$ | **48.28 ± 0.59** |

---

## 📁 Repository Structure

```
.
├── final_ieee_paper/          # Complete IEEE Transactions LaTeX manuscript & figures
│   ├── main.tex               # 10-page IEEE journal manuscript
│   ├── supplementary_material.tex # Standalone 2-page supplementary material
│   ├── references.bib         # Fully audited BibTeX bibliography
│   ├── figures/               # 100% native LaTeX/TikZ vector graphics (Figs. 1–5)
│   ├── final_ieee_manuscript.pdf # Compiled submission-ready PDF (10 pages)
│   └── supplementary_material.pdf # Compiled supplementary PDF (2 pages)
├── configs/                   # Experiment and benchmark configuration files
├── ph_ssd/                    # Core PyTorch source code
│   ├── modules/               # HEDO coordinate-momentum & HVSC latent modules
│   ├── backbones/             # Chunk-wise SSD state-continuous recurrence
│   ├── models/                # End-to-end multimodal dual-stream architecture
│   ├── losses/                # Multi-positive InfoNCE & symmetric KL losses
│   └── training/              # Multi-seed trainer & checkpoint manager
├── train.py                   # Model training entrypoint
├── evaluate.py                # Standalone evaluation & metric computation
├── benchmark.py               # 12-run factorial multi-seed benchmark runner
└── README.md
```

---

## 🛠️ Execution & Reproducibility

### 1. Environment Setup
```bash
conda create -n hedo_hvsc python=3.12 -y
conda activate hedo_hvsc
pip install torch torchvision timm transformers
```

### 2. Run Multi-Seed Benchmark
```bash
python benchmark.py --config configs/config.yaml --seeds 42 43 44
```

### 3. Evaluate Checkpoints
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

---

## 📜 Citation & License

This project is licensed under the Apache 2.0 License.
