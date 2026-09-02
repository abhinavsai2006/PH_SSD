"""
VL-Mamba Baseline Architecture (Qiao et al., 2024).
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from ph_ssd.backbones.ssd_wrapper import SSDBlock


class BaselineVLMamba(nn.Module):
    """
    VL-Mamba Baseline Model.
    Uses linear projector and joint sequence scan without SD-NPF energy dissipation.
    """

    def __init__(self, input_dim: int = 768, d_model: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.proj_v = nn.Linear(input_dim, d_model)
        self.proj_t = nn.Linear(input_dim, d_model)
        self.ssd = SSDBlock(d_model=d_model, d_state=64)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, raw_A: torch.Tensor, raw_B: torch.Tensor) -> Dict[str, torch.Tensor]:
        v = self.proj_v(raw_A)
        t = self.proj_t(raw_B)
        x = torch.cat([v, t], dim=1)
        h, _ = self.ssd(x)
        logits = self.classifier(h.mean(dim=1))

        return {
            "logits": logits,
            "kl_loss": torch.tensor(0.0, device=x.device),
        }
