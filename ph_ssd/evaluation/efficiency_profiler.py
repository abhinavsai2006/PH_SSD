"""
GPU Computational Efficiency Profiler (Latency, Throughput, Peak VRAM, FLOPs, Params).
Author: Lead Research Engineer
License: Apache 2.0
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def profile_model_efficiency(
    model: nn.Module,
    sample_input_A: torch.Tensor,
    sample_input_B: torch.Tensor,
    warmup_iters: int = 5,
    test_iters: int = 20,
) -> Dict[str, Any]:
    """
    Profile model execution latency, throughput, peak GPU memory, FLOPs, and parameter counts.

    Args:
        model (nn.Module): PyTorch model instance.
        sample_input_A (torch.Tensor): Dummy Modality A input.
        sample_input_B (torch.Tensor): Dummy Modality B input.

    Returns:
        Dict[str, Any]: Profile metrics dictionary.
    """
    model.eval()
    device = next(model.parameters()).device

    sample_input_A = sample_input_A.to(device)
    sample_input_B = sample_input_B.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(sample_input_A, sample_input_B)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Measure latency & throughput
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(sample_input_A, sample_input_B)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / test_iters) * 1000.0
    throughput_fps = (test_iters * sample_input_A.size(0)) / total_time

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Estimate FLOPs (approximate: 2 * Params * Seq_Len)
    seq_len = sample_input_A.size(1)
    estimated_gflops = (2.0 * total_params * seq_len) / 1e9

    return {
        "params/total_m": total_params / 1e6,
        "params/trainable_m": trainable_params / 1e6,
        "efficiency/latency_ms": avg_latency_ms,
        "efficiency/throughput_fps": throughput_fps,
        "efficiency/peak_memory_mb": peak_memory_mb,
        "efficiency/estimated_gflops": estimated_gflops,
    }
