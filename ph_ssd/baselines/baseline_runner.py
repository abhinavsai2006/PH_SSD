"""
Automated Baseline Benchmark Comparison Suite Runner.
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from ph_ssd.datasets.synthetic_multimodal import SyntheticMultimodalDataset
from ph_ssd.models.ph_ssd_model import PHSSDTaskModel
from ph_ssd.baselines.baseline_mamba2 import BaselineMamba2
from ph_ssd.baselines.baseline_vl_mamba import BaselineVLMamba
from ph_ssd.baselines.baseline_clip import BaselineCLIP
from ph_ssd.evaluation.evaluator import PHSSDEvaluator


class BaselineRunner:
    """
    Automated Baseline Comparison Runner.
    Compares PH-SSD against Baseline Mamba-2, VL-Mamba, and CLIP.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device: torch.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_all_baselines(self) -> Dict[str, Dict[str, float]]:
        dataset = SyntheticMultimodalDataset(num_samples=40, seq_len=32, dim_A=64, dim_B=64, num_classes=5)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        models = {
            "PH-SSD (Proposed)": PHSSDTaskModel(input_dim_A=64, input_dim_B=64, d_model=64, num_classes=5),
            "Baseline Mamba-2": BaselineMamba2(input_dim=64, d_model=64, num_classes=5),
            "VL-Mamba": BaselineVLMamba(input_dim=64, d_model=64, num_classes=5),
            "CLIP Dual-Encoder": BaselineCLIP(input_dim=64, d_model=64, num_classes=5),
        }

        results: Dict[str, Dict[str, float]] = {}

        for name, model in models.items():
            evaluator = PHSSDEvaluator(model, device=self.device)
            res = evaluator.evaluate(dataloader)
            results[name] = res

        return results
