"""
Custom Multimodal & Signal Analysis Metrics.
Author: Senior AI Systems Architect
License: Apache 2.0
"""

import torch


def compute_snr_improvement(
    input_signal: torch.Tensor, filtered_signal: torch.Tensor
) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) improvement in decibels (dB).

    Args:
        input_signal (torch.Tensor): Unfiltered noisy signal tensor.
        filtered_signal (torch.Tensor): Output filtered signal tensor.

    Returns:
        float: SNR improvement in dB.
    """
    signal_power = torch.sum(filtered_signal ** 2)
    noise_power = torch.sum((input_signal - filtered_signal) ** 2) + 1e-8
    snr_db = 10.0 * torch.log10(signal_power / noise_power)
    return snr_db.item()


def compute_effective_rank(feature_matrix: torch.Tensor) -> float:
    """
    Compute Effective Rank (Roy & Vetterli, 2007) of feature matrix singular spectrum.

    Args:
        feature_matrix (torch.Tensor): Feature matrix tensor of shape (N, D)

    Returns:
        float: Effective rank scalar.
    """
    if feature_matrix.dim() > 2:
        feature_matrix = feature_matrix.reshape(-1, feature_matrix.size(-1))

    # Singular Value Decomposition
    _, S, _ = torch.linalg.svd(feature_matrix.float(), full_matrices=False)
    p = S / (torch.sum(S) + 1e-8)
    entropy = -torch.sum(p * torch.log(p + 1e-8))
    eff_rank = torch.exp(entropy)
    return eff_rank.item()
