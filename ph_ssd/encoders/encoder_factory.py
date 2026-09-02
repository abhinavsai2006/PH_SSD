"""
Encoder Factory Method for Vision and Text Backbone Instantiation.
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch.nn as nn
from typing import Tuple

from ph_ssd.encoders.vision_encoders import PretrainedVisionEncoder
from ph_ssd.encoders.text_encoders import PretrainedTextEncoder


def build_multimodal_encoders(
    input_dim_A: int = 768,
    input_dim_B: int = 768,
    vision_model: str = "vit",
    text_model: str = "roberta",
    embed_dim: int = 768,
    pretrained: bool = False,
) -> Tuple[nn.Module, nn.Module]:
    """
    Factory function to instantiate Vision and Text Encoders.

    Args:
        input_dim_A (int): Modality A input feature dimension.
        input_dim_B (int): Modality B input feature dimension.
        vision_model (str): Vision architecture ('siglip', 'dinov2', 'vit').
        text_model (str): Text architecture ('roberta', 'deberta', 'bert').
        embed_dim (int): Common target embedding dimension.
        pretrained (bool): Whether to load weights from HuggingFace/timm.

    Returns:
        Tuple[nn.Module, nn.Module]: (vision_encoder, text_encoder)
    """
    v_enc = PretrainedVisionEncoder(input_dim=input_dim_A, model_name=vision_model, embed_dim=embed_dim, pretrained=pretrained)
    t_enc = PretrainedTextEncoder(input_dim=input_dim_B, model_name=text_model, embed_dim=embed_dim, pretrained=pretrained)
    return v_enc, t_enc
