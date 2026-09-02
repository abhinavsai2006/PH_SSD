"""
Dataset Manager for Multimodal Benchmark Verification.
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
from typing import Dict, Any, List


class DatasetManager:
    """
    Dataset Manager for inspecting and verifying multimodal datasets on disk.
    Supported Datasets: Flickr30k, MS COCO Karpathy Split, VQA v2, TextVQA, DocVQA.
    """

    SUPPORTED_DATASETS: List[str] = [
        "flickr30k",
        "mscoco",
        "vqav2",
        "textvqa",
        "docvqa",
    ]

    def __init__(self, data_root: str = "data") -> None:
        self.data_root: str = data_root

    def verify_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Verify status of dataset on local filesystem.

        Args:
            dataset_name (str): Name of dataset.

        Returns:
            Dict[str, Any]: Verification status dictionary.
        """
        name = dataset_name.lower().strip()
        dataset_dir = os.path.join(self.data_root, name)

        if not os.path.exists(dataset_dir):
            return {
                "dataset": name,
                "status": "MISSING",
                "directory": dataset_dir,
                "images_count": 0,
                "annotations_count": 0,
                "train_samples": 0,
                "val_samples": 0,
                "test_samples": 0,
                "missing_files": [f"Directory '{dataset_dir}' does not exist."],
            }

        missing_files = []
        images_count = 0
        annotations_count = 0

        if name == "flickr30k":
            img_dir = os.path.join(dataset_dir, "flickr30k_images")
            ann_file = os.path.join(dataset_dir, "results_20130124.token")

            if not os.path.exists(img_dir):
                missing_files.append(f"Missing images directory: '{img_dir}'")
            else:
                images_count = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

            if not os.path.exists(ann_file):
                missing_files.append(f"Missing annotations file: '{ann_file}'")
            else:
                with open(ann_file, "r", encoding="utf-8") as f:
                    annotations_count = sum(1 for _ in f)

        elif name == "mscoco":
            ann_file = os.path.join(dataset_dir, "dataset_coco.json")
            if not os.path.exists(ann_file):
                missing_files.append(f"Missing Karpathy split file: '{ann_file}'")

        elif name == "vqav2":
            q_file = os.path.join(dataset_dir, "v2_OpenEnded_mscoco_train2014_questions.json")
            if not os.path.exists(q_file):
                missing_files.append(f"Missing VQA v2 questions file: '{q_file}'")

        status = "FOUND" if len(missing_files) == 0 and images_count > 0 else "MISSING"

        return {
            "dataset": name,
            "status": status,
            "directory": dataset_dir,
            "images_count": images_count,
            "annotations_count": annotations_count,
            "train_samples": images_count,
            "val_samples": 0,
            "test_samples": 0,
            "missing_files": missing_files,
        }

    def verify_all(self) -> Dict[str, Dict[str, Any]]:
        """Verify status of all supported datasets."""
        return {ds: self.verify_dataset(ds) for ds in self.SUPPORTED_DATASETS}
