"""
CLIP Baseline Dual-Encoder Architecture.
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class BaselineCLIP(nn.Module):
    """
    Standard Dual-Encoder Baseline (CLIP-style).
    Uses separate vision and text encoders without cross-modal sequence scan.
    """

    def __init__(self, input_dim: int = 768, d_model: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.enc_v = nn.Linear(input_dim, d_model)
        self.enc_t = nn.Linear(input_dim, d_model)
        self.classifier = nn.Linear(2 * d_model, num_classes)

    def forward(self, raw_A: torch.Tensor, raw_B: torch.Tensor) -> Dict[str, torch.Tensor]:
        v_pool = self.enc_v(raw_A).mean(dim=1)
        t_pool = self.enc_t(raw_B).mean(dim=1)
        joint = torch.cat([v_pool, t_pool], dim=-1)
        logits = self.classifier(joint)

        return {
            "logits": logits,
            "kl_loss": torch.tensor(0.0, device=raw_A.device),
        }
