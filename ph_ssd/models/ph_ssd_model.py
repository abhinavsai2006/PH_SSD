"""
End-to-End Multimodal PH-SSD Contrastive Architecture.
Author: Lead Research Engineer
License: Apache 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional

from ph_ssd.encoders.encoder_factory import build_multimodal_encoders
from ph_ssd.backbones.multimodal_ph_ssd import MultimodalPHSSDBackbone


class PHSSDTaskModel(nn.Module):
    """
    End-to-End PH-SSD Multimodal Representation Architecture for Image-Text Contrastive Retrieval.

    Integrates:
      - Pretrained/Built-in Encoders (Vision & Text)
      - SD-NPF Spatial Energy Dissipation Pre-Filter
      - Mamba-2 / SSD State-Space Dualities Sequence Scan
      - VCM-SSD Closed-Form Variational Chunk Boundary Coupling
      - Cross-Modal L2-Normalized Embedding Alignment Heads with Learnable Temperature

    Ablation Configurations:
      - Baseline Mamba/SSD: use_sd_npf=False, use_vcm_ssd=False
      - Mamba/SSD + SD-NPF:  use_sd_npf=True,  use_vcm_ssd=False
      - Mamba/SSD + VCM-SSD: use_sd_npf=False, use_vcm_ssd=True
      - Full PH-SSD:         use_sd_npf=True,  use_vcm_ssd=True
    """

    def __init__(
        self,
        input_dim_A: int = 768,
        input_dim_B: int = 768,
        d_model: int = 128,
        d_embed: int = 128,
        d_state: int = 64,
        z_dim: int = 32,
        n_layers: int = 2,
        vision_encoder: str = "vit",
        text_encoder: str = "roberta",
        pretrained: bool = False,
        use_sd_npf: bool = True,
        use_vcm_ssd: bool = True,
        init_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.input_dim_A: int = input_dim_A
        self.input_dim_B: int = input_dim_B
        self.d_model: int = d_model
        self.d_embed: int = d_embed
        self.use_sd_npf: bool = use_sd_npf
        self.use_vcm_ssd: bool = use_vcm_ssd

        # Encoders for Modality A (Vision) and Modality B (Text)
        self.encoder_A, self.encoder_B = build_multimodal_encoders(
            input_dim_A=input_dim_A,
            input_dim_B=input_dim_B,
            vision_model=vision_encoder,
            text_model=text_encoder,
            embed_dim=d_model,
            pretrained=pretrained,
        )

        # PH-SSD Backbone with Ablation Toggles
        self.backbone: MultimodalPHSSDBackbone = MultimodalPHSSDBackbone(
            d_model=d_model,
            d_state=d_state,
            z_dim=z_dim,
            n_layers=n_layers,
            use_sd_npf=use_sd_npf,
            use_vcm_ssd=use_vcm_ssd,
        )

        # Projection heads into joint multimodal embedding space
        self.proj_embed_A = nn.Sequential(
            nn.Linear(d_model, d_embed),
            nn.LayerNorm(d_embed),
        )
        self.proj_embed_B = nn.Sequential(
            nn.Linear(d_model, d_embed),
            nn.LayerNorm(d_embed),
        )

        # Learnable log-temperature parameter for CLIP-style InfoNCE similarity scaling
        self.log_temperature = nn.Parameter(torch.ones([]) * math.log(1.0 / init_temperature))

    def forward(
        self, raw_A: torch.Tensor, raw_B: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward multimodal pass outputting normalized embeddings for contrastive alignment.

        Args:
            raw_A (torch.Tensor): Visual input sequence (Batch, Seq_Len, input_dim_A)
            raw_B (torch.Tensor): Text input sequence (Batch, Seq_Len, input_dim_B)

        Returns:
            Dict[str, torch.Tensor]: Output dictionary containing:
                - "embed_A": Normalized vision embedding (Batch, d_embed)
                - "embed_B": Normalized text embedding (Batch, d_embed)
                - "logit_scale": Temperature scaling multiplier
                - "kl_loss": Cross-modal KL divergence loss scalar
                - "energy_tracks": Stacked energy tracks tensor
        """
        x_A = self.encoder_A(raw_A)
        x_B = self.encoder_B(raw_B)

        out_A, out_B, kl_loss, energy_tracks = self.backbone(x_A, x_B)

        pool_A = out_A.mean(dim=1)  # (Batch, d_model)
        pool_B = out_B.mean(dim=1)  # (Batch, d_model)

        embed_A = F.normalize(self.proj_embed_A(pool_A), p=2, dim=-1)
        embed_B = F.normalize(self.proj_embed_B(pool_B), p=2, dim=-1)

        logit_scale = torch.clamp(self.log_temperature.exp(), max=100.0)

        energy_tensor = torch.stack(energy_tracks, dim=0)

        return {
            "embed_A": embed_A,
            "embed_B": embed_B,
            "logit_scale": logit_scale,
            "kl_loss": kl_loss,
            "energy_tracks": energy_tensor,
        }
