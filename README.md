# Port-Hamiltonian State Space Dualities (PH-SSD)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)]()

Publication-grade research codebase implementing **Port-Hamiltonian State Space Dualities (PH-SSD)** for long-sequence multimodal representation learning.

---

## Core Modules

1. **Symplectic Dissipative Neural Pre-Filter (SD-NPF):**  
   Symplectic Euler state-space pre-filter enforcing continuous energy decay ($\dot{H} \le -p^T C p \le 0$) to attenuate background noise (`ph_ssd/modules/sd_npf.py`).

2. **Variational Cross-Modal SSD Boundary Coupling (VCM-SSD):**  
   Closed-form Gaussian Variational Information Bottleneck executing at SSD block boundaries ($B=64$), preventing modality collapse (`ph_ssd/modules/vcm_ssd.py`).

3. **Official Mamba-2 Core:**  
   Integrates `mamba_ssm` CUDA scan kernels with high-performance vector-fused PyTorch fallback (`ph_ssd/backbones/official_mamba2.py`).

---

## Clean Directory Structure

```
.
├── configs/                   # Production YAML configuration files
│   ├── config.yaml            # Master model & dataset config
│   ├── compact_flickr8k.yaml  # Lightweight local config (~1 GB storage)
│   ├── retrieval_flickr30k.yaml
│   ├── retrieval_mscoco.yaml
│   └── vqa_v2.yaml
├── scripts/                   # Dataset management & CLI tools
│   ├── download_dataset.py    # Master downloader CLI
│   ├── verify_dataset.py      # Dataset verifier & status inspector
│   ├── extract_dataset.py     # Zip/Tar archive extractor
│   ├── checksum.py            # SHA-256 / MD5 checksum verifier
│   ├── download_utils.py      # HTTP Range download with resume
│   └── dataset_status.py      # Terminal status table generator
├── ph_ssd/                    # Core PyTorch package
│   ├── modules/               # SD-NPF & VCM-SSD core modules
│   ├── backbones/             # Mamba-2 & Multimodal PH-SSD backbone
│   ├── encoders/              # Vision (ViT, DINOv2, SigLIP) & Text (RoBERTa, DeBERTa)
│   ├── models/                # End-to-end task architecture
│   ├── losses/                # Joint task & VIB KL divergence loss
│   ├── datasets/              # Dataset loaders (Flickr8k, Flickr30k, COCO, VQA v2, Custom)
│   ├── training/              # AMP Trainer, EMA, Cosine Warmup Scheduler
│   ├── evaluation/            # Retrieval, generation & efficiency metrics
│   ├── utils/                 # Logger, Checkpoint Manager & LaTeX Exporters
│   ├── visualization/         # Energy trajectory & result plotting
│   └── tests/                 # Comprehensive PyTest unit test suite
├── train.py                   # Training entrypoint
├── evaluate.py                # Evaluation & efficiency profiler entrypoint
├── benchmark.py               # Baseline comparison & ablation matrix runner
├── infer.py                   # Inference API service entrypoint
├── export_torchscript.py      # TorchScript JIT tracing exporter
└── export_onnx.py             # ONNX model exporter
```

---

## Quick Start & Workflows

### 1. Run Unit Tests
```bash
py -m unittest discover -s ph_ssd/tests
```

### 2. Dataset Downloader & Verification
```bash
# Setup Flickr8k lightweight dataset (~1.0 GB)
py -m scripts.download_dataset --dataset flickr8k

# Inspect dataset readiness status
py -m scripts.verify_dataset --dataset all
```

### 3. Local Training & Evaluation
```bash
# Train on Flickr8k lightweight benchmark
py train.py --config configs/compact_flickr8k.yaml

# Evaluate model checkpoint & profile hardware efficiency
py evaluate.py --config configs/compact_flickr8k.yaml
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
