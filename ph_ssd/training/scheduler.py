"""
Cosine LR Scheduler with Warmup.
Author: Lead Research Engineer
License: Apache 2.0
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupLRScheduler(_LRScheduler):
    """
    Cosine Annealing Learning Rate Scheduler with Linear Warmup.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs: int = warmup_epochs
        self.total_epochs: int = total_epochs
        self.min_lr: float = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear Warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]

        # Cosine Annealing
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]
