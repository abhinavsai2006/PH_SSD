"""
Exponential Moving Average (EMA) Model Utility.
Author: Lead Research Engineer
License: Apache 2.0
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Any


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) of model parameter weights.

    Attributes:
        decay (float): Decay factor coefficient (e.g. 0.999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay: float = decay
        self.ema_model: nn.Module = copy.deepcopy(model).eval()

        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters using current model weights."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, Any]:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.ema_model.load_state_dict(state_dict)
