"""
Multimodal PH-SSD Backbone Architecture with Ablation Toggles.
Author: Senior AI Systems Architect
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from ph_ssd.modules.sd_npf import SymplecticDissipativeNeuralPreFilter
from ph_ssd.modules.vcm_ssd import VariationalCrossModalSSDCoupler
from ph_ssd.backbones.ssd_wrapper import SSDBlock


class MultimodalPHSSDBackbone(nn.Module):
    """
    Multimodal PH-SSD Backbone with support for ablation experiments:
      - Exp A: Baseline Mamba/SSD (use_sd_npf=False, use_vcm_ssd=False)
      - Exp B: Mamba/SSD + SD-NPF (use_sd_npf=True, use_vcm_ssd=False)
      - Exp C: Mamba/SSD + VCM-SSD (use_sd_npf=False, use_vcm_ssd=True)
      - Exp D: Full PH-SSD (use_sd_npf=True, use_vcm_ssd=True)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        z_dim: int = 32,
        n_layers: int = 2,
        chunk_size: int = 64,
        delta_t: float = 0.1,
        beta: float = 1e-3,
        use_sd_npf: bool = True,
        use_vcm_ssd: bool = True,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.d_state: int = d_state
        self.n_layers: int = n_layers
        self.chunk_size: int = chunk_size
        self.use_sd_npf: bool = use_sd_npf
        self.use_vcm_ssd: bool = use_vcm_ssd

        # Modality A & B Pre-Filters (SD-NPF)
        self.npf_A = nn.ModuleList([SymplecticDissipativeNeuralPreFilter(d_model, delta_t) for _ in range(n_layers)])
        self.npf_B = nn.ModuleList([SymplecticDissipativeNeuralPreFilter(d_model, delta_t) for _ in range(n_layers)])

        # Modality A & B SSD Scan Blocks
        self.ssd_A = nn.ModuleList([SSDBlock(d_model, d_state, chunk_size) for _ in range(n_layers)])
        self.ssd_B = nn.ModuleList([SSDBlock(d_model, d_state, chunk_size) for _ in range(n_layers)])

        # Cross-Modal Variational Couplers (VCM-SSD)
        self.vcm_couplers = nn.ModuleList([VariationalCrossModalSSDCoupler(d_state, z_dim, beta) for _ in range(n_layers)])

    def forward(
        self, x_A: torch.Tensor, x_B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if x_A.size(1) != x_B.size(1):
            # Align sequence length dimension across modalities if needed (e.g. interpolate x_B to match x_A length)
            min_seq = min(x_A.size(1), x_B.size(1))
            x_A = x_A[:, :min_seq, :]
            x_B = x_B[:, :min_seq, :]


        total_kl_loss = torch.tensor(0.0, device=x_A.device, dtype=x_A.dtype)
        energy_tracks = []

        h_A_boundary: Optional[torch.Tensor] = None
        h_B_boundary: Optional[torch.Tensor] = None

        h_A_curr = x_A
        h_B_curr = x_B

        for i in range(self.n_layers):
            # Step 1: Pre-filtering with SD-NPF (if enabled)
            if self.use_sd_npf:
                h_A_filtered, energy_A = self.npf_A[i](h_A_curr)
                h_B_filtered, energy_B = self.npf_B[i](h_B_curr)
                energy_tracks.append(energy_A)
            else:
                h_A_filtered = h_A_curr
                h_B_filtered = h_B_curr
                energy_tracks.append(torch.zeros(x_A.size(0), x_A.size(1), device=x_A.device))

            # Step 2: SSD intra-chunk parallel scan
            out_A, h_A_next_state = self.ssd_A[i](h_A_filtered, initial_state=h_A_boundary)
            out_B, h_B_next_state = self.ssd_B[i](h_B_filtered, initial_state=h_B_boundary)

            # Step 3: VCM-SSD boundary coupling (if enabled)
            if self.use_vcm_ssd:
                h_A_boundary, h_B_boundary, kl_loss = self.vcm_couplers[i](h_A_next_state, h_B_next_state)
                total_kl_loss = total_kl_loss + kl_loss
            else:
                h_A_boundary = h_A_next_state
                h_B_boundary = h_B_next_state

            h_A_curr = out_A
            h_B_curr = out_B

        return h_A_curr, h_B_curr, total_kl_loss, energy_tracks
