"""
Dataset Factory Module for Multimodal Representation Learning.
Supports full datasets and lightweight reproducible subsets (Flickr8k, Flickr30k, MS COCO, VQA v2, Custom).
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
from torch.utils.data import Dataset
from ph_ssd.datasets.synthetic_multimodal import SyntheticMultimodalDataset
from ph_ssd.datasets.flickr30k import Flickr30kDataset
from ph_ssd.datasets.flickr8k import Flickr8kDataset
from ph_ssd.datasets.mscoco import MSCOCODataset
from ph_ssd.datasets.vqav2 import VQAv2Dataset
from ph_ssd.datasets.custom_multimodal import CustomMultimodalDomainDataset


def build_dataset(
    dataset_name: str = "flickr8k",
    data_dir: str = "data",
    split: str = "train",
    seq_len: int = 64,
    image_size: int = 224,
    max_samples: int = 0,
) -> Dataset:
    """
    Build dataset instance based on dataset_name.

    Args:
        dataset_name (str): Name of dataset ('flickr8k', 'flickr30k', 'mscoco', 'vqav2', 'custom', 'synthetic').
        data_dir (str): Root directory for datasets.
        split (str): 'train', 'val', or 'test'.
        seq_len (int): Sequence length.
        image_size (int): Image size.
        max_samples (int): Max samples to load for quick demo / benchmarking (0 = full dataset).

    Returns:
        Dataset: Instantiated PyTorch Dataset object.
    """
    name = dataset_name.lower().strip()

    def get_path(ds_sub: str) -> str:
        if data_dir.endswith(ds_sub):
            return data_dir
        return os.path.join(data_dir, ds_sub)

    if name == "synthetic":
        return SyntheticMultimodalDataset(seq_len=seq_len, num_samples=max_samples or 100)
    elif name == "flickr8k":
        return Flickr8kDataset(data_dir=get_path("flickr8k"), split=split, seq_len=seq_len, image_size=image_size, max_samples=max_samples)
    elif name in ["flickr30k_subset", "flickr30k_small", "flickr30k"]:
        return Flickr30kDataset(data_dir=get_path("flickr30k"), split=split, seq_len=seq_len, image_size=image_size)
    elif name == "mscoco":
        return MSCOCODataset(data_dir=get_path("mscoco"), split=split, seq_len=seq_len, image_size=image_size)
    elif name == "vqav2":
        return VQAv2Dataset(data_dir=get_path("vqav2"), split=split, seq_len=seq_len, image_size=image_size)
    elif name in ["custom", "custom_multimodal"]:
        return CustomMultimodalDomainDataset(data_dir=get_path("custom_multimodal"), split=split, seq_len=seq_len, image_size=image_size)
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Supported options: ['flickr8k', 'flickr30k', 'mscoco', 'vqav2', 'custom', 'synthetic']."
        )
