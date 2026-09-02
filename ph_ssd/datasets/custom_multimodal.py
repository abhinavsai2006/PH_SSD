"""
Custom Domain Multimodal Dataset Loader (~1-2 GB).
Supports custom image-text-question domain datasets for lightweight research benchmarking.
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Any, List

from ph_ssd.datasets.tokenization import MultimodalPreprocessor


class CustomMultimodalDomainDataset(Dataset):
    """
    Custom Multimodal Domain Dataset Loader (~1–2 GB).

    Reads custom images and JSON annotation files (`annotations.json`).
    JSON Schema Expected:
    [
        {"image": "img_001.jpg", "caption": "A custom image description.", "label": 0},
        ...
    ]

    Raises:
        FileNotFoundError: If custom dataset directory or JSON annotations are missing.
    """

    def __init__(
        self,
        data_dir: str = "data/custom_multimodal",
        split: str = "train",
        seq_len: int = 64,
        image_size: int = 224,
        max_samples: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir: str = data_dir
        self.split: str = split
        self.seq_len: int = seq_len
        self.preprocessor: MultimodalPreprocessor = MultimodalPreprocessor(image_size=image_size, max_text_len=seq_len)

        self.images_dir = os.path.join(data_dir, "images")
        self.annotations_file = os.path.join(data_dir, "annotations.json")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Custom multimodal directory '{data_dir}' not found. "
                f"Create directory '{data_dir}', add image files in '{self.images_dir}', and 'annotations.json' annotations file."
            )

        self.samples: List[Dict[str, Any]] = []
        if os.path.exists(self.annotations_file) and os.path.exists(self.images_dir):
            with open(self.annotations_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    img_path = os.path.join(self.images_dir, item["image"])
                    if os.path.exists(img_path):
                        self.samples.append({
                            "image_path": img_path,
                            "caption": item.get("caption", "Sample caption"),
                            "label": item.get("label", 0),
                        })

        if max_samples > 0 and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No valid custom multimodal pairs found in '{self.data_dir}'. "
                f"Ensure images exist in '{self.images_dir}' and 'annotations.json' is properly formatted."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        img_tensor = self.preprocessor.preprocess_image(image)

        img_seq = img_tensor.unfold(1, 16, 16).unfold(2, 16, 16).permute(1, 2, 0, 3, 4).reshape(-1, 768)
        if img_seq.size(0) > self.seq_len:
            img_seq = img_seq[:self.seq_len]
        elif img_seq.size(0) < self.seq_len:
            pad = torch.zeros(self.seq_len - img_seq.size(0), 768)
            img_seq = torch.cat([img_seq, pad], dim=0)

        text_tokens = self.preprocessor.tokenize_text_simple(sample["caption"], vocab_size=768)
        text_seq = torch.eye(768)[text_tokens % 768]

        return {
            "raw_A": img_seq,
            "raw_B": text_seq,
            "target": torch.tensor(sample["label"], dtype=torch.long),
        }
