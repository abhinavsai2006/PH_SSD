"""
Publication-Quality Visualization Plotter Suite.
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional


def plot_training_curves(
    train_losses: List[float], val_accuracies: List[float], save_path: str = "docs/training_curves.png"
) -> str:
    """Plot publication loss and accuracy curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(train_losses, color="tab:red", label="Training Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)", color="tab:blue")
    ax2.plot(val_accuracies, color="tab:blue", label="Val Accuracy", linewidth=2, linestyle="--")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("PH-SSD Multimodal Training Dynamics")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_ablation_bar_chart(
    results: Dict[str, Dict[str, float]], save_path: str = "docs/ablation_chart.png"
) -> str:
    """Plot publication ablation study bar chart."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names = list(results.keys())
    accuracies = [res.get("eval/accuracy", 0.0) for res in results.values()]

    plt.figure(figsize=(9, 5))
    bars = plt.barh(names, accuracies, color=["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"])
    plt.xlabel("Evaluation Accuracy (%)")
    plt.title("Component Ablation Study Breakdown")
    plt.xlim(0, 100)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1.0, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", ha="left", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
