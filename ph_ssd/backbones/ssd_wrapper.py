"""
Mamba-2 State-Space Duality (SSD) Block Wrapper.
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ph_ssd.backbones.official_mamba2 import OfficialMamba2Block


class SSDBlock(nn.Module):
    """
    Mamba-2 State-Space Duality (SSD) Block Wrapper.

    Delegates to OfficialMamba2Block for official CUDA integration or fallback execution.
    """

    def __init__(self, d_model: int, d_state: int = 64, chunk_size: int = 64, require_native_mamba: bool = True) -> None:
        super().__init__()
        self.block = OfficialMamba2Block(
            d_model=d_model,
            d_state=d_state,
            chunk_size=chunk_size,
            require_native_mamba=require_native_mamba,
        )

    def forward(
        self, x: torch.Tensor, initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.block(x, initial_state)
