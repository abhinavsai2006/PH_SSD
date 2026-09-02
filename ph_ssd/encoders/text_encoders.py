"""
Pretrained Text Encoders (RoBERTa, DeBERTa-v3, BERT).
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Optional

HAS_TRANSFORMERS = False
try:
    from transformers import AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class PretrainedTextEncoder(nn.Module):
    """
    Pretrained Text Encoder Wrapper supporting RoBERTa, DeBERTa-v3, and BERT.
    """

    def __init__(
        self,
        input_dim: int = 768,
        model_name: str = "roberta",
        embed_dim: int = 768,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.model_name: str = model_name.lower().strip()
        self.embed_dim: int = embed_dim

        if HAS_TRANSFORMERS:
            hf_name = "roberta-base"
            if "deberta" in self.model_name:
                hf_name = "microsoft/deberta-v3-base"
            elif "bert" in self.model_name:
                hf_name = "bert-base-uncased"
            self.backbone = AutoModel.from_pretrained(hf_name)
            self.proj = nn.Linear(self.backbone.config.hidden_size, embed_dim)
        else:
            raise ImportError("transformers library is required to load pretrained Text Encoders.")

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward text feature extraction pass.

        Args:
            x (torch.Tensor): Token ID tensor (Batch, Seq_Len) or sequence tensor (Batch, Seq_Len, Feature_Dim)
            attention_mask (Optional[torch.Tensor]): Attention mask tensor (Batch, Seq_Len)

        Returns:
            torch.Tensor: Projected text sequence tensor (Batch, Seq_Len, embed_dim)
        """
        if x.dtype == torch.long or x.dim() == 2:
            outputs = self.backbone(input_ids=x, attention_mask=attention_mask)
            return self.proj(outputs.last_hidden_state)

        return self.proj(x)

