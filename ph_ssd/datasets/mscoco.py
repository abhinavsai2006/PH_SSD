"""
MS COCO Karpathy Split Multimodal Dataset Loader.
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Any, List

from ph_ssd.datasets.tokenization import MultimodalPreprocessor


class MSCOCODataset(Dataset):
    """
    MS COCO Karpathy Split Image-Caption Dataset Loader.

    Raises:
        FileNotFoundError: If MS COCO dataset images or annotations are missing.
    """

    def __init__(
        self,
        data_dir: str = "data/mscoco",
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
                f"MS COCO dataset directory '{data_dir}' not found. "
                f"Please download MS COCO Karpathy split images and annotations into '{data_dir}' using 'python scripts/download_dataset.py --dataset mscoco'."
            )

        self.samples: List[Dict[str, Any]] = []
        annotations_file = os.path.join(data_dir, f"dataset_coco.json")

        if not os.path.exists(annotations_file):
            raise FileNotFoundError(
                f"MS COCO Karpathy annotations file '{annotations_file}' not found. "
                f"Ensure 'dataset_coco.json' exists inside '{data_dir}'."
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
