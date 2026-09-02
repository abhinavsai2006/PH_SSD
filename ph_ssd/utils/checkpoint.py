"""
Model Checkpoint Manager.
Author: Senior AI Systems Architect
License: Apache 2.0
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class CheckpointManager:
    """
    Model Checkpoint Manager for saving and resuming training state.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        filename: str = "checkpoint_latest.pt",
    ) -> str:
        """
        Save model and optimizer state to disk.

        Args:
            model (nn.Module): PyTorch model.
            optimizer (torch.optim.Optimizer): PyTorch optimizer.
            epoch (int): Current epoch.
            filename (str): Output filename.

        Returns:
            str: Path to saved checkpoint file.
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(state, filepath)
        return filepath

    def load_checkpoint(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> int:
        """
        Load model and optimizer state from disk checkpoint.

        Args:
            filepath (str): Path to checkpoint file.
            model (nn.Module): Target model instance.
            optimizer (Optional[torch.optim.Optimizer]): Target optimizer instance.

        Returns:
            int: Epoch index restored from checkpoint.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]
