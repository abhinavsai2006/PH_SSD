"""
PH-SSD Joint Symmetric Cross-Modal Contrastive Loss Function.
Author: Senior AI Systems Architect
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class PHSSDLoss(nn.Module):
    """
    PH-SSD Joint Symmetric Cross-Modal Objective Function.

    Combines:
      1. Symmetric InfoNCE Cross-Modal Contrastive Loss (Image <-> Text Alignment).
      2. VCM-SSD Cross-Modal Variational KL-Divergence Loss.

    Formula:
      L_contrastive = 0.5 * (CrossEntropy(logits_i2t, targets) + CrossEntropy(logits_t2i, targets))
      L_total = contrastive_weight * L_contrastive + kl_weight * L_kl
    """

    def __init__(self, contrastive_weight: float = 1.0, kl_weight: float = 1e-3) -> None:
        super().__init__()
        self.contrastive_weight: float = contrastive_weight
        self.kl_weight: float = kl_weight
        self.cross_entropy: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute joint contrastive training loss.

        Args:
            outputs (Dict[str, torch.Tensor]): Output dictionary from PHSSDTaskModel containing:
                - "embed_A": L2-normalized image embeddings (Batch, d_embed)
                - "embed_B": L2-normalized text embeddings (Batch, d_embed)
                - "logit_scale": Temperature scaling factor
                - "kl_loss": VCM-SSD KL divergence loss scalar
            targets (Optional[torch.Tensor]): Optional targets (unused or for custom alignment).

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - total_loss: Weighted scalar loss for autograd
                - metrics: Loggable metrics dictionary
        """
        embed_A = outputs["embed_A"]
        embed_B = outputs["embed_B"]
        logit_scale = outputs["logit_scale"]
        kl_loss = outputs["kl_loss"]

        batch_size = embed_A.size(0)
        device = embed_A.device

        # Cosine similarity matrix scaled by temperature: S_ij = (e_v,i . e_t,j) * logit_scale
        sim_matrix = logit_scale * torch.matmul(embed_A, embed_B.t())  # (Batch, Batch)
        labels = torch.arange(batch_size, device=device, dtype=torch.long)

        # Image-to-Text and Text-to-Image Cross-Entropy Contrastive Loss
        loss_i2t = self.cross_entropy(sim_matrix, labels)
        loss_t2i = self.cross_entropy(sim_matrix.t(), labels)
        contrastive_loss = 0.5 * (loss_i2t + loss_t2i)

        # Total Loss with VCM-SSD Variational Information Bottleneck KL Regularization
        total_loss = self.contrastive_weight * contrastive_loss + self.kl_weight * kl_loss

        metrics = {
            "loss/total": total_loss.item(),
            "loss/contrastive": contrastive_loss.item(),
            "loss/i2t": loss_i2t.item(),
            "loss/t2i": loss_t2i.item(),
            "loss/kl": kl_loss.item(),
        }

        return total_loss, metrics
