"""
Production-Grade PyTorch Contrastive Trainer for PH-SSD.
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple

from ph_ssd.models.ph_ssd_model import PHSSDTaskModel
from ph_ssd.losses.ph_ssd_loss import PHSSDLoss
from ph_ssd.training.ema import ExponentialMovingAverage
from ph_ssd.training.scheduler import CosineWarmupLRScheduler


class PHSSDTrainer:
    """
    Production-Grade PH-SSD Model Contrastive Trainer.

    Supports:
      - Symmetric InfoNCE Cross-Modal Contrastive Training
      - Mixed Precision Training (AMP)
      - Exponential Moving Average (EMA)
      - Cosine Learning Rate Scheduler with Warmup
      - Gradient Clipping & Checkpointing
    """

    def __init__(
        self,
        model: PHSSDTaskModel,
        criterion: PHSSDLoss,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        use_ema: bool = True,
        max_grad_norm: float = 1.0,
        warmup_epochs: int = 1,
        total_epochs: int = 10,
    ) -> None:
        self.model: PHSSDTaskModel = model
        self.criterion: PHSSDLoss = criterion
        self.optimizer: torch.optim.Optimizer = optimizer
        self.device: torch.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp: bool = use_amp and torch.cuda.is_available()
        self.max_grad_norm: float = max_grad_norm

        self.model.to(self.device)
        self.scaler: torch.amp.GradScaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.ema: Optional[ExponentialMovingAverage] = ExponentialMovingAverage(model) if use_ema else None
        self.scheduler: CosineWarmupLRScheduler = CosineWarmupLRScheduler(
            optimizer, warmup_epochs=warmup_epochs, total_epochs=total_epochs
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Execute one training epoch.

        Args:
            dataloader (DataLoader): Training DataLoader.

        Returns:
            Dict[str, float]: Aggregated metric dictionary.
        """
        self.model.train()
        total_loss_acc = 0.0
        contrastive_loss_acc = 0.0
        kl_loss_acc = 0.0
        total_samples = 0

        for batch in dataloader:
            raw_A = batch["raw_A"].to(self.device)
            raw_B = batch["raw_B"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(raw_A, raw_B)
                loss, metrics = self.criterion(outputs)

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            batch_size = raw_A.size(0)
            total_loss_acc += loss.item() * batch_size
            contrastive_loss_acc += metrics["loss/contrastive"] * batch_size
            kl_loss_acc += metrics["loss/kl"] * batch_size
            total_samples += batch_size

        self.scheduler.step()

        return {
            "epoch/loss": total_loss_acc / total_samples,
            "epoch/contrastive_loss": contrastive_loss_acc / total_samples,
            "epoch/kl_loss": kl_loss_acc / total_samples,
            "epoch/lr": self.optimizer.param_groups[0]["lr"],
        }
