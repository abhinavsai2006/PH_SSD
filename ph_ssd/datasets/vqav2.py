"""
VQA v2 Visual Question Answering Dataset Loader.
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Any, List

from ph_ssd.datasets.tokenization import MultimodalPreprocessor


class VQAv2Dataset(Dataset):
    """
    VQA v2 Dataset Loader.

    Raises:
        FileNotFoundError: If VQA v2 dataset files are missing.
    """

    def __init__(
        self,
        data_dir: str = "data/vqav2",
        split: str = "train",
        seq_len: int = 64,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.data_dir: str = data_dir
        self.split: str = split
        self.seq_len: int = seq_len
        self.preprocessor: MultimodalPreprocessor = MultimodalPreprocessor(image_size=image_size, max_text_len=seq_len)

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"VQA v2 dataset directory '{data_dir}' not found. "
                f"Please download VQA v2 images and questions into '{data_dir}' using 'python scripts/download_dataset.py --dataset vqav2'."
            )

        self.samples: List[Dict[str, Any]] = []

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

        text_tokens = self.preprocessor.tokenize_text_simple(sample["question"], vocab_size=768)
        text_seq = torch.eye(768)[text_tokens % 768]

        return {
            "raw_A": img_seq,
            "raw_B": text_seq,
            "target": torch.tensor(sample["label"], dtype=torch.long),
        }
