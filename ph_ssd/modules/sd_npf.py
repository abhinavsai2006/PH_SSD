"""
Symplectic Dissipative Neural Pre-Filter (SD-NPF)
Author: Senior AI Systems Architect
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SymplecticDissipativeNeuralPreFilter(nn.Module):
    """
    Symplectic Dissipative Neural Pre-Filter (SD-NPF).

    Applies discrete Symplectic Euler integration over learned feature energy manifolds
    to attenuate background feature noise prior to State Space Model scanning.

    Mathematical Formulation:
        p_t = (I - Δt C) * p_{t-1} - Δt K * q_{t-1} + Δt * W_x * x_t
        q_t = q_{t-1} + Δt * p_t
        x̂_t = W_out * tanh(q_t)

    Attributes:
        d_model (int): Hidden feature dimension.
        delta_t (float): Integration step size.
    """

    def __init__(self, d_model: int, delta_t: float = 0.1) -> None:
        """
        Initialize SD-NPF module.

        Args:
            d_model (int): Feature dimension.
            delta_t (float): Discrete integration step size. Default: 0.1.
        """
        super().__init__()
        self.d_model: int = d_model
        self.delta_t: float = delta_t

        # Learnable diagonal parameters: C > 0 (dissipation), K > 0 (stiffness)
        self.theta_c: nn.Parameter = nn.Parameter(torch.zeros(d_model))
        self.theta_k: nn.Parameter = nn.Parameter(torch.zeros(d_model))

        # Projection weights
        self.W_x: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.W_out: nn.Linear = nn.Linear(d_model, d_model, bias=False)

        # Initialize orthogonal matrices for isometric state mapping
        nn.init.orthogonal_(self.W_x.weight)
        nn.init.orthogonal_(self.W_out.weight)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward execution pass.

        Args:
            x (torch.Tensor): Input sequence tensor of shape (Batch, Seq_Len, d_model)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_hat: Filtered output sequence tensor of shape (Batch, Seq_Len, d_model)
                - H_energy: Hamiltonian energy track tensor of shape (Batch, Seq_Len)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (B, N, D), got shape {x.shape}")
        B, N, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Dimension mismatch: expected {self.d_model}, got {D}")

        device, dtype = x.device, x.dtype

        # Enforce positive definiteness: C > 0, K > 0 with clamping for FP16/BF16 stability
        C: torch.Tensor = torch.exp(torch.clamp(self.theta_c, min=-5.0, max=3.0))
        K: torch.Tensor = torch.exp(torch.clamp(self.theta_k, min=-5.0, max=3.0))

        # Initialize phase space state vectors: q_0 = 0, p_0 = 0
        q_t: torch.Tensor = torch.zeros(B, D, device=device, dtype=dtype)
        p_t: torch.Tensor = torch.zeros(B, D, device=device, dtype=dtype)

        x_proj: torch.Tensor = self.W_x(x)

        q_steps = []
        energy_steps = []

        decay_factor: torch.Tensor = 1.0 - self.delta_t * C

        for t in range(N):
            x_t = x_proj[:, t, :]

            # Symplectic Euler Recurrence Step
            p_t = decay_factor * p_t - self.delta_t * K * q_t + self.delta_t * x_t
            q_t = q_t + self.delta_t * p_t

            # Compute Hamiltonian Energy: H(q, p) = 0.5 * ||p||^2 + 0.5 * q^T K q
            H_t: torch.Tensor = 0.5 * torch.sum(p_t ** 2, dim=-1) + 0.5 * torch.sum(K * (q_t ** 2), dim=-1)

            q_steps.append(q_t)
            energy_steps.append(H_t)

        q_stack: torch.Tensor = torch.stack(q_steps, dim=1)        # (B, N, D)
        energy_stack: torch.Tensor = torch.stack(energy_steps, dim=1)  # (B, N)

        x_hat: torch.Tensor = self.W_out(torch.tanh(q_stack))

        return x_hat, energy_stack
