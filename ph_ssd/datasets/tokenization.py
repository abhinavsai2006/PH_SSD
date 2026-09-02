"""
Multimodal Image Preprocessing & Text Tokenization Utilities.
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
from torchvision import transforms
from PIL import Image
from typing import Dict, Any, List, Optional, Union


class MultimodalPreprocessor:
    """
    Image and Text Preprocessing Utility for Multimodal Encoders.
    """

    def __init__(self, image_size: int = 224, max_text_len: int = 64) -> None:
        self.image_size: int = image_size
        self.max_text_len: int = max_text_len

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image into standard normalized tensor.

        Args:
            image (Image.Image): Input PIL RGB image.

        Returns:
            torch.Tensor: Preprocessed image tensor of shape (3, image_size, image_size)
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.image_transform(image)

    def tokenize_text_simple(self, text: str, vocab_size: int = 30522) -> torch.Tensor:
        """
        Tokenize raw string into token ID tensor (simple hashing tokenizer fallback).

        Args:
            text (str): Input text caption/question.
            vocab_size (int): Vocabulary limit size.

        Returns:
            torch.Tensor: Token IDs tensor of shape (max_text_len,)
        """
        words = text.lower().strip().split()
        tokens = [hash(w) % (vocab_size - 2) + 2 for w in words[:self.max_text_len]]

        # Pad or truncate to max_text_len
        if len(tokens) < self.max_text_len:
            tokens = tokens + [0] * (self.max_text_len - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)
