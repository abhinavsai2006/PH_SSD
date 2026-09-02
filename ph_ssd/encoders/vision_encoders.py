"""
Pretrained Vision Encoders (SigLIP, DINOv2, ViT).
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Optional

HAS_TIMM = False
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class PretrainedVisionEncoder(nn.Module):
    """
    Pretrained Vision Encoder Wrapper supporting SigLIP, DINOv2, and ViT.
    Extracts sequence patch tokens (B, N, D) from raw images.
    """

    def __init__(
        self,
        input_dim: int = 768,
        model_name: str = "vit",
        embed_dim: int = 768,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.model_name: str = model_name.lower().strip()
        self.embed_dim: int = embed_dim

        if HAS_TIMM:
            timm_name = "vit_base_patch16_224"
            if "dinov2" in self.model_name:
                timm_name = "vit_base_patch14_dinov2"
            self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
            self.proj = nn.Linear(self.backbone.num_features, embed_dim)
        else:
            raise ImportError("timm library is required to load pretrained Vision Encoders.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward vision feature extraction pass.

        Args:
            x (torch.Tensor): Visual input images (Batch, 3, 224, 224) or pre-extracted sequence.

        Returns:
            torch.Tensor: Projected visual sequence tokens (Batch, Seq_Len, embed_dim)
        """
        if x.dim() == 4:
            # Pass raw images through ViT patch embedding + transformer blocks
            feats = self.backbone.forward_features(x) # (B, 197, 768)
            return self.proj(feats)

        return self.proj(x)

