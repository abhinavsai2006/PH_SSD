"""
Variational Cross-Modal SSD Boundary Coupling (VCM-SSD)
Author: Senior AI Systems Architect
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VariationalCrossModalSSDCoupler(nn.Module):
    """
    Variational Cross-Modal SSD Boundary Coupler (VCM-SSD).

    Computes closed-form Gaussian Variational Information Bottleneck interactions
    at SSD chunk boundaries for dual-modality state continuity without breaking intra-chunk scan.

    Attributes:
        d_state (int): State Space Model hidden state dimension.
        z_dim (int): Latent bottleneck dimension.
        beta (float): KL-divergence penalty weighting coefficient.
    """

    def __init__(self, d_state: int, z_dim: int, beta: float = 1e-3) -> None:
        """
        Initialize VCM-SSD module.

        Args:
            d_state (int): Hidden state dimension.
            z_dim (int): Variational latent space dimension.
            beta (float): Weighting coefficient for KL divergence loss. Default: 1e-3.
        """
        super().__init__()
        self.d_state: int = d_state
        self.z_dim: int = z_dim
        self.beta: float = beta

        # Joint Encoder: concatenates boundary states [h_A, h_B] -> mu, logvar
        self.encoder: nn.Sequential = nn.Sequential(
            nn.Linear(2 * d_state, d_state),
            nn.SiLU(),
            nn.Linear(d_state, 2 * z_dim)
        )

        # State Injectors
        self.W_A: nn.Linear = nn.Linear(z_dim, d_state, bias=False)
        self.W_B: nn.Linear = nn.Linear(z_dim, d_state, bias=False)

        # Initialize state injectors
        nn.init.kaiming_normal_(self.W_A.weight)
        nn.init.kaiming_normal_(self.W_B.weight)


    def forward(
        self, h_A: torch.Tensor, h_B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward execution pass across chunk boundaries.

        Args:
            h_A (torch.Tensor): Boundary state for Modality A (Batch, d_state)
            h_B (torch.Tensor): Boundary state for Modality B (Batch, d_state)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - h_A_next: Coupled initial state for Modality A (Batch, d_state)
                - h_B_next: Coupled initial state for Modality B (Batch, d_state)
                - kl_loss: Closed-form KL divergence loss scalar
        """
        if h_A.shape != h_B.shape:
            raise ValueError(f"State shape mismatch: {h_A.shape} vs {h_B.shape}")
        if h_A.shape[-1] != self.d_state:
            raise ValueError(f"Expected d_state {self.d_state}, got {h_A.shape[-1]}")

        h_joint: torch.Tensor = torch.cat([h_A, h_B], dim=-1)
        stats: torch.Tensor = self.encoder(h_joint)

        mu, logvar = torch.chunk(stats, 2, dim=-1)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        std: torch.Tensor = torch.exp(0.5 * logvar)

        # Reparameterization Trick
        eps: torch.Tensor = torch.randn_like(std)
        z: torch.Tensor = mu + eps * std

        # Compute coupled states
        h_A_next: torch.Tensor = h_A + self.W_A(z)
        h_B_next: torch.Tensor = h_B + self.W_B(z)

        # Closed-form Gaussian KL Divergence: KL(N(mu, sigma) || N(0, I))
        kl_loss: torch.Tensor = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        return h_A_next, h_B_next, self.beta * kl_loss
