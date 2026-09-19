"""Frozen pretrained backbones (ViT-B/16 and RoBERTa-base).

Both are kept in eval mode with requires_grad=False. Their outputs are cached
once by features.py; training never re-runs them.
"""

import os

import torch
import torch.nn as nn
import torchvision.models as tv_models

VIT_WEIGHTS = tv_models.ViT_B_16_Weights.IMAGENET1K_V1
ROBERTA_NAME = "roberta-base"
# Pin a Hugging Face commit for exact reproducibility. "main" is resolved and the
# concrete commit hash is recorded in feature_manifest.json.
ROBERTA_REVISION = os.environ.get("HEDO_ROBERTA_REVISION", "main")
MAX_TEXT_LEN = 64


def vit_transform():
    """Official preprocessing for the chosen weights (Resize 256 -> CenterCrop 224 -> Normalize)."""
    return VIT_WEIGHTS.transforms()


class FrozenViT(nn.Module):
    """torchvision VisionTransformer.forward without the class-token selection and head.

    Returns the 196 patch tokens after the encoder's final LayerNorm. The encoder
    adds pos_embedding internally.
    """

    def __init__(self):
        super().__init__()
        self.vit = tv_models.vit_b_16(weights=VIT_WEIGHTS)
        assert self.vit.encoder.pos_embedding.shape == (1, 197, 768)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        x = self.vit._process_input(x)
        cls = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = self.vit.encoder(torch.cat([cls, x], dim=1))
        return x[:, 1:, :]

    @torch.no_grad()
    def forward_with_saliency(self, x):
        """Same outputs as forward(), plus last-block class-token attention to the 196 patches (head-averaged).

        Used only as a foreground/background proxy for the H1 probe; never as a model input.
        """
        enc = self.vit.encoder
        x = self.vit._process_input(x)
        x = torch.cat([self.vit.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = enc.dropout(x + enc.pos_embedding)
        for layer in enc.layers[:-1]:
            x = layer(x)
        last = enc.layers[-1]
        y = last.ln_1(x)
        _, attn = last.self_attention(y, y, y, need_weights=True, average_attn_weights=True)
        x = enc.ln(last(x))
        return x[:, 1:, :], attn[:, 0, 1:]


class FrozenRoBERTa(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.roberta, info = AutoModel.from_pretrained(
            ROBERTA_NAME, revision=ROBERTA_REVISION, add_pooling_layer=False, output_loading_info=True)
        # lm_head.* appear as unexpected keys: the checkpoint is an MLM checkpoint and
        # the LM head is never instantiated. Any *missing* key is a real problem.
        self.loading_info = {k: sorted(v) if isinstance(v, (list, set)) else v for k, v in info.items()}
        if self.loading_info.get("missing_keys"):
            raise RuntimeError(f"RoBERTa missing weights: {self.loading_info['missing_keys']}")
        unexpected = [k for k in self.loading_info.get("unexpected_keys", []) if not k.startswith("lm_head.")]
        if unexpected:
            raise RuntimeError(f"RoBERTa unexpected non-LM-head weights: {unexpected}")
        self.commit_hash = getattr(self.roberta.config, "_commit_hash", None)
        for p in self.roberta.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        return self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROBERTA_NAME, revision=ROBERTA_REVISION)


def tokenize(tokenizer, captions):
    return tokenizer(list(captions), padding="max_length", truncation=True, max_length=MAX_TEXT_LEN,
                     return_tensors="pt")
