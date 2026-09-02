"""
Energy Dissipation Track Visualization.
Author: Senior AI Systems Architect
License: Apache 2.0
"""

import matplotlib.pyplot as plt
import torch
from typing import List, Optional


def plot_energy_dissipation(
    energy_tracks: List[torch.Tensor], save_path: Optional[str] = None
) -> None:
    """
    Plot Hamiltonian energy trajectory across sequence tokens for SD-NPF layers.

    Args:
        energy_tracks (List[torch.Tensor]): List of energy track tensors of shape (Batch, Seq_Len)
        save_path (Optional[str]): Optional filepath to save output plot PNG.
    """
    plt.figure(figsize=(10, 5))

    for idx, track in enumerate(energy_tracks):
        # Mean across batch dimension
        mean_energy = track.mean(dim=0).detach().cpu().numpy()
        plt.plot(mean_energy, label=f"SD-NPF Layer {idx + 1}")

    plt.title("Port-Hamiltonian Monotonic Energy Dissipation Trajectory")
    plt.xlabel("Sequence Token Index (t)")
    plt.ylabel("Hamiltonian Energy H(q_t, p_t)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
