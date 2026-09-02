"""
Automatic Dataset Setup and Downloader Utility.
Supports Flickr8k (~1 GB), Flickr30k, MS COCO, VQA v2, TextVQA, DocVQA, and Custom Datasets.
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
import urllib.request
from typing import Optional


def download_file(url: str, dest_path: str) -> str:
    """
    Download file from URL to destination path.

    Args:
        url (str): Source URL.
        dest_path (str): File destination path.

    Returns:
        str: Path to downloaded file.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"Downloading {url} -> {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    return dest_path


def setup_flickr8k_data(data_dir: str = "data/flickr8k") -> str:
    """Print download instructions and setup directory for lightweight Flickr8k (~1 GB)."""
    os.makedirs(data_dir, exist_ok=True)
    print(f"=== Flickr8k Lightweight Dataset Setup (~1 GB) ===")
    print(f"Directory: {os.path.abspath(data_dir)}")
    print(f"1. Open one of the active Kaggle Flickr8k download links in your browser:")
    print(f"   - Primary: https://www.kaggle.com/datasets/adityajn105/flickr8k")
    print(f"   - Alternative: https://www.kaggle.com/datasets/srbhshinde/flickr8k-dataset")
    print(f"2. Log into Kaggle and click 'Download' (zip file ~1.0 GB).")
    print(f"3. Extract archive contents into: {os.path.abspath(data_dir)}")
    print(f"4. Ensure images are in '{os.path.join(data_dir, 'Images')}' and captions in 'captions.txt' or 'Flickr8k.token.txt'.")
    return data_dir


def setup_flickr30k_data(data_dir: str = "data/flickr30k") -> str:
    """Print download instructions and setup directory for Flickr30k."""
    os.makedirs(data_dir, exist_ok=True)
    print(f"=== Flickr30k Dataset Setup ===")
    print(f"Directory: {os.path.abspath(data_dir)}")
    print(f"1. Download Flickr30k images (31,783 images) from Kaggle: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset")
    print(f"2. Extract images into: {os.path.join(data_dir, 'flickr30k_images')}")
    print(f"3. Place 'results_20130124.token' annotations inside: {data_dir}")
    return data_dir


def setup_mscoco_data(data_dir: str = "data/mscoco") -> str:
    """Print download instructions and setup directory for MS COCO Karpathy Split."""
    os.makedirs(data_dir, exist_ok=True)
    print(f"=== MS COCO Karpathy Split Setup ===")
    print(f"Directory: {os.path.abspath(data_dir)}")
    print(f"1. Download COCO train2014 & val2014 images from: https://cocodataset.org/#download")
    print(f"2. Download Karpathy split 'dataset_coco.json' from: https://cs.stanford.edu/people/karpathy/deepimagesent/")
    print(f"3. Place 'dataset_coco.json' inside: {data_dir}")
    return data_dir


def setup_vqav2_data(data_dir: str = "data/vqav2") -> str:
    """Print download instructions and setup directory for VQA v2."""
    os.makedirs(data_dir, exist_ok=True)
    print(f"=== VQA v2 Setup ===")
    print(f"Directory: {os.path.abspath(data_dir)}")
    print(f"1. Download VQA v2 train questions and annotations from: https://visualqa.org/download.html")
    print(f"2. Place questions and annotations JSON files inside: {data_dir}")
    return data_dir
