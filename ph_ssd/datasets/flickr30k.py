"""
Flickr30k Multimodal Dataset Loader.
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Any, List, Optional

from ph_ssd.datasets.tokenization import MultimodalPreprocessor


class Flickr30kDataset(Dataset):
    """
    Flickr30k Image-Text Dataset Loader.

    Loads official Flickr30k dataset images and captions for multimodal representation
    learning and retrieval.

    Raises:
        FileNotFoundError: If the Flickr30k image or caption directory does not exist.
    """

    def __init__(
        self,
        data_dir: str = "data/flickr30k",
        split: str = "train",
        seq_len: int = 64,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.data_dir: str = data_dir
        self.split: str = split
        self.seq_len: int = seq_len
        self.preprocessor: MultimodalPreprocessor = MultimodalPreprocessor(image_size=image_size, max_text_len=seq_len)

        self.images_dir = os.path.join(data_dir, "flickr30k_images")
        self.annotations_file = os.path.join(data_dir, "results_20130124.token")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Flickr30k dataset directory '{data_dir}' not found. "
                f"Please download official Flickr30k images and captions into '{data_dir}' using 'python scripts/download_dataset.py --dataset flickr30k'."
            )

        self.samples: List[Dict[str, Any]] = []
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        img_id = parts[0].split("#")[0]
                        caption = parts[1]
                        img_path = os.path.join(self.images_dir, img_id)
                        if os.path.exists(img_path):
                            self.samples.append({
                                "image_path": img_path,
                                "caption": caption,
                                "label": len(self.samples) % 10,
                            })

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No valid image-caption pairs found in '{self.data_dir}'. "
                f"Ensure Flickr30k images exist in '{self.images_dir}' and annotations in '{self.annotations_file}'."
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
