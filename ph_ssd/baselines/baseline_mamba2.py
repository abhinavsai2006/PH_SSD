"""
Standard Baseline Mamba-2 Model (Unimodal Recurrence).
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from ph_ssd.backbones.ssd_wrapper import SSDBlock


class BaselineMamba2(nn.Module):
    """
    Standard Baseline Mamba-2 Architecture.
    Processes concatenated multimodal sequence without SD-NPF or VCM-SSD coupling.
    """

    def __init__(self, input_dim: int = 768, d_model: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.ssd1 = SSDBlock(d_model=d_model, d_state=64)
        self.ssd2 = SSDBlock(d_model=d_model, d_state=64)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, raw_A: torch.Tensor, raw_B: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Simple concatenation along sequence dimension
        x = torch.cat([raw_A, raw_B], dim=1)
        h = self.proj(x)
        h, _ = self.ssd1(h)
        h, _ = self.ssd2(h)
        pool = h.mean(dim=1)
        logits = self.classifier(pool)

        return {
            "logits": logits,
            "kl_loss": torch.tensor(0.0, device=x.device),
        }
