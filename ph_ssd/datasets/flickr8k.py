"""
Flickr8k Multimodal Dataset Loader with Reproducible Train/Val/Test Split.
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Any, List, Set, Tuple

from ph_ssd.datasets.tokenization import MultimodalPreprocessor


class Flickr8kDataset(Dataset):
    """
    Flickr8k Multimodal Dataset Loader with strict non-overlapping Train/Val/Test Splits.
    Preserves multi-caption structure per image identity.
    """

    def __init__(
        self,
        data_dir: str = "data/flickr8k",
        split: str = "train",
        seq_len: int = 64,
        image_size: int = 224,
        max_samples: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir: str = data_dir
        self.split: str = split.lower().strip()
        self.seq_len: int = seq_len
        self.preprocessor: MultimodalPreprocessor = MultimodalPreprocessor(image_size=image_size, max_text_len=seq_len)

        # Check image directory path
        self.images_dir = os.path.join(data_dir, "Images")
        if not os.path.exists(self.images_dir):
            self.images_dir = os.path.join(data_dir, "flickr8k_images")
        if not os.path.exists(self.images_dir):
            self.images_dir = data_dir

        # Check annotation file path
        self.annotations_file = os.path.join(data_dir, "captions.txt")
        if not os.path.exists(self.annotations_file):
            self.annotations_file = os.path.join(data_dir, "Flickr8k.token.txt")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Flickr8k dataset directory '{data_dir}' not found.")

        # Read raw annotation pairs
        raw_pairs: List[Tuple[str, str, int]] = []
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, "r", encoding="utf-8") as f:
                header = True
                cap_counters: Dict[str, int] = {}
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    if header and ("image" in line_str.lower() and "caption" in line_str.lower()):
                        header = False
                        continue
                    header = False
                    if "," in line_str:
                        parts = line_str.split(",", 1)
                        img_id = parts[0].strip()
                        caption = parts[1].strip()
                    elif "\t" in line_str:
                        parts = line_str.split("\t")
                        img_id = parts[0].split("#")[0].strip()
                        caption = parts[1].strip()
                    else:
                        continue

                    if not caption or len(caption) < 2:
                        continue

                    img_path = os.path.join(self.images_dir, img_id)
                    if os.path.exists(img_path):
                        cap_idx = cap_counters.get(img_id, 0)
                        cap_counters[img_id] = cap_idx + 1
                        raw_pairs.append((img_id, caption, cap_idx))

        # Perform deterministic split by unique image IDs to prevent data leakage
        unique_image_ids = sorted(list(set(pair[0] for pair in raw_pairs)))
        rng = random.Random(seed)
        shuffled_ids = list(unique_image_ids)
        rng.shuffle(shuffled_ids)

        n_total_imgs = len(shuffled_ids)
        n_train = int(n_total_imgs * 0.80)  # 6,472 images
        n_val = int(n_total_imgs * 0.10)    # 809 images

        train_imgs: Set[str] = set(shuffled_ids[:n_train])
        val_imgs: Set[str] = set(shuffled_ids[n_train:n_train + n_val])
        test_imgs: Set[str] = set(shuffled_ids[n_train + n_val:])

        if self.split in ["train", "training"]:
            target_imgs = train_imgs
        elif self.split in ["val", "validation"]:
            target_imgs = val_imgs
        elif self.split in ["test", "testing"]:
            target_imgs = test_imgs
        else:
            target_imgs = set(unique_image_ids)

        # Filter samples for assigned split
        self.samples: List[Dict[str, Any]] = []
        for img_id, caption, cap_idx in raw_pairs:
            if img_id in target_imgs:
                img_path = os.path.join(self.images_dir, img_id)
                self.samples.append({
                    "image_id": img_id,
                    "caption_id": f"{img_id}#{cap_idx}",
                    "image_path": img_path,
                    "caption": caption,
                })

        if max_samples > 0 and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        # Export split report if in root directory
        os.makedirs("paper_results", exist_ok=True)
        split_report = {
            "dataset_name": "Flickr8k",
            "total_raw_pairs": len(raw_pairs),
            "unique_images": n_total_imgs,
            "train_images": len(train_imgs),
            "val_images": len(val_imgs),
            "test_images": len(test_imgs),
            "current_split": self.split,
            "samples_in_split": len(self.samples),
        }
        with open("paper_results/dataset_split.json", "w", encoding="utf-8") as f:
            json.dump(split_report, f, indent=2)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        img_tensor = self.preprocessor.preprocess_image(image)

        # Patch unfolding into (196, 768) visual token sequence
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
            "image_id": sample["image_id"],
            "caption_id": sample["caption_id"],
            "caption": sample["caption"],
        }

