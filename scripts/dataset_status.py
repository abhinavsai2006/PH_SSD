"""
Multimodal Benchmark Dataset Status & Integrity Inspection Tool.
Author: Lead MLOps Engineer
License: Apache 2.0
"""

import os
from typing import Dict, Any, List


SUPPORTED_DATASETS: List[str] = [
    "flickr8k",
    "flickr30k",
    "mscoco",
    "vqav2",
    "textvqa",
    "docvqa",
    "nocaps",
    "vizwiz",
    "refcoco",
    "refcoco+",
    "refcocog",
    "custom",
]


def inspect_dataset_status(data_root: str = "data") -> Dict[str, Dict[str, Any]]:
    """
    Inspect filesystem status for all supported multimodal datasets.

    Args:
        data_root (str): Root data folder.

    Returns:
        Dict[str, Dict[str, Any]]: Status mapping dictionary.
    """
    results = {}
    for ds in SUPPORTED_DATASETS:
        if ds == "refcoco+":
            folder_name = "refcoco_plus"
        elif ds == "custom":
            folder_name = "custom_multimodal"
        else:
            folder_name = ds

        ds_path = os.path.join(data_root, folder_name)

        exists = os.path.exists(ds_path)
        images_count = 0
        annotations_count = 0
        disk_size_mb = 0.0

        if exists:
            for root, _, files in os.walk(ds_path):
                for f in files:
                    fp = os.path.join(root, f)
                    disk_size_mb += os.path.getsize(fp) / (1024 * 1024)
                    if f.lower().endswith((".jpg", ".png", ".jpeg")):
                        images_count += 1
                    elif f.lower().endswith((".json", ".token", ".txt", ".pth")):
                        annotations_count += 1

        status = "READY" if (exists and images_count > 0) else "MISSING"

        results[ds] = {
            "dataset": ds,
            "status": status,
            "images": images_count,
            "annotations": annotations_count,
            "size_mb": round(disk_size_mb, 2),
            "directory": ds_path,
        }

    return results


def print_status_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print formatted terminal report table."""
    print("=" * 80)
    print(f"{'Dataset Name':<14} | {'Status':<8} | {'Image Count':<12} | {'Anns Count':<12} | {'Disk Size (MB)':<14}")
    print("=" * 80)

    all_ready = True
    for ds_name, info in results.items():
        st = info["status"]
        img = info["images"]
        ann = info["annotations"]
        sz = info["size_mb"]

        print(f"{ds_name:<14} | {st:<8} | {img:<12} | {ann:<12} | {sz:<14}")
        if st != "READY":
            all_ready = False

    print("=" * 80)
    if all_ready:
        print("\n[OVERALL VERDICT] ALL DATASETS READY FOR MULTIMODAL TRAINING.")
    else:
        print("\n[OVERALL VERDICT] NOTICE — ONE OR MORE DATASETS ARE MISSING.")


if __name__ == "__main__":
    res = inspect_dataset_status("data")
    print_status_table(res)
