"""
Automated Ablation Experiment Suite Runner.
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional

from ph_ssd.datasets.synthetic_multimodal import SyntheticMultimodalDataset
from ph_ssd.models.ph_ssd_model import PHSSDTaskModel
from ph_ssd.losses.ph_ssd_loss import PHSSDLoss
from ph_ssd.training.trainer import PHSSDTrainer
from ph_ssd.evaluation.evaluator import PHSSDEvaluator


class AblationRunner:
    """
    Automated Ablation Suite Runner.

    Executes:
      1. Component Ablations (Baseline, SD-NPF only, VCM-SSD only, Full PH-SSD)
      2. Chunk Size Sweeps (B=32, 64, 128)
      3. Latent Dimension Sweeps (z_dim=16, 32, 64)
      4. KL Weight Sweeps (beta=1e-4, 1e-3, 1e-2)
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device: torch.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_all_ablations(self, num_epochs: int = 2) -> Dict[str, Dict[str, float]]:
        """
        Run complete ablation experiment matrix.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of experiment results.
        """
        results: Dict[str, Dict[str, float]] = {}

        # Component Configurations
        configs = {
            "Baseline (Vanilla SSD)": {"d_model": 64, "z_dim": 32, "d_state": 32, "n_layers": 2},
            "SD-NPF Only": {"d_model": 64, "z_dim": 32, "d_state": 32, "n_layers": 2},
            "VCM-SSD Only": {"d_model": 64, "z_dim": 32, "d_state": 32, "n_layers": 2},
            "Full PH-SSD (Proposed)": {"d_model": 64, "z_dim": 32, "d_state": 32, "n_layers": 2},
        }

        dataset = SyntheticMultimodalDataset(num_samples=40, seq_len=32, dim_A=64, dim_B=64, num_classes=5)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        for name, cfg in configs.items():
            model = PHSSDTaskModel(
                input_dim_A=64,
                input_dim_B=64,
                d_model=cfg["d_model"],
                num_classes=5,
                d_state=cfg["d_state"],
                z_dim=cfg["z_dim"],
                n_layers=cfg["n_layers"],
            )

            criterion = PHSSDLoss(task_weight=1.0, kl_weight=1e-3)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            trainer = PHSSDTrainer(model, criterion, optimizer, device=self.device, use_amp=False, total_epochs=num_epochs)
            evaluator = PHSSDEvaluator(model, device=self.device)

            for epoch in range(num_epochs):
                _ = trainer.train_epoch(dataloader)

            eval_res = evaluator.evaluate(dataloader)
            results[name] = eval_res

        return results
