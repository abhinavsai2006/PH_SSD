"""
Master Multimodal Benchmark Dataset Downloader & Setup System.
Supports Flickr8k (~1 GB), Flickr30k, MS COCO 2014, VQA v2, TextVQA, DocVQA, NoCaps, VizWiz, RefCOCO, RefCOCO+, RefCOCOg, Custom.
Author: Lead MLOps Engineer
License: Apache 2.0
"""

import os
import argparse
from scripts.download_utils import download_with_resume, extract_archive
from scripts.dataset_status import inspect_dataset_status, print_status_table


DATASET_SOURCES = {
    "flickr8k": {
        "name": "Flickr8k Lightweight Dataset (~1 GB)",
        "primary_url": "https://www.kaggle.com/datasets/adityajn105/flickr8k",
        "alt_url": "https://www.kaggle.com/datasets/srbhshinde/flickr8k-dataset",
        "dest": "data/flickr8k",
        "expected_images": 8000,
        "instructions": (
            "1. Open active Kaggle dataset link:\n"
            "   Primary:     https://www.kaggle.com/datasets/adityajn105/flickr8k\n"
            "   Alternative: https://www.kaggle.com/datasets/srbhshinde/flickr8k-dataset\n"
            "2. Log in and click 'Download' (~1.0 GB zip archive).\n"
            "3. Extract archive contents into 'data/flickr8k/'.\n"
            "4. Ensure images are in 'data/flickr8k/Images/' and text annotations in 'captions.txt' or 'Flickr8k.token.txt'."
        ),
    },
    "flickr30k": {
        "name": "Flickr30k Image-Text Dataset",
        "url": "https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset",
        "dest": "data/flickr30k",
        "expected_images": 31783,
        "instructions": (
            "1. Visit: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset\n"
            "2. Download archive into 'data/flickr30k/'\n"
            "3. Extract images into 'data/flickr30k/flickr30k_images/'\n"
            "4. Place 'results_20130124.token' in 'data/flickr30k/'"
        ),
    },
    "mscoco": {
        "name": "MS COCO 2014 Karpathy Split",
        "train_img_url": "http://images.cocodataset.org/zips/train2014.zip",
        "val_img_url": "http://images.cocodataset.org/zips/val2014.zip",
        "karpathy_url": "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip",
        "dest": "data/mscoco",
        "expected_images": 123287,
        "instructions": (
            "1. Automatic download available for images & split JSON.\n"
            "2. Images saved to 'data/mscoco/train2014/' and 'val2014/'\n"
            "3. Annotations saved to 'data/mscoco/dataset_coco.json'"
        ),
    },
    "vqav2": {
        "name": "VQA v2 Visual Question Answering",
        "train_q_url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        "train_ann_url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "dest": "data/vqav2",
        "instructions": "Automatic download available for question and annotation files into 'data/vqav2/'.",
    },
    "textvqa": {
        "name": "TextVQA Dataset",
        "url": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
        "dest": "data/textvqa",
        "instructions": "Automatic download available from Facebook AI Research into 'data/textvqa/'.",
    },
    "docvqa": {
        "name": "DocVQA Document Visual Question Answering",
        "url": "https://rrc.cvc.uab.es/?ch=17",
        "dest": "data/docvqa",
        "instructions": (
            "1. Visit official RRC portal: https://rrc.cvc.uab.es/?ch=17\n"
            "2. Download document images & train_v1.0.json into 'data/docvqa/'"
        ),
    },
    "nocaps": {
        "name": "NoCaps Novel Object Captioning",
        "url": "https://nocaps.org/download",
        "dest": "data/nocaps",
        "instructions": "Download evaluation images and annotations from https://nocaps.org/download into 'data/nocaps/'.",
    },
    "vizwiz": {
        "name": "VizWiz Visual QA for Blind Users",
        "url": "https://vizwiz.org/tasks-and-datasets/vqa/",
        "dest": "data/vizwiz",
        "instructions": "Download train/val images and annotations from https://vizwiz.org into 'data/vizwiz/'.",
    },
    "refcoco": {
        "name": "RefCOCO Visual Grounding",
        "url": "https://github.com/licota/refer",
        "dest": "data/refcoco",
        "instructions": "Download Refer API datasets from https://github.com/licota/refer into 'data/refcoco/'.",
    },
    "refcoco+": {
        "name": "RefCOCO+ Visual Grounding",
        "url": "https://github.com/licota/refer",
        "dest": "data/refcoco_plus",
        "instructions": "Download Refer API datasets into 'data/refcoco_plus/'.",
    },
    "refcocog": {
        "name": "RefCOCOg Visual Grounding",
        "url": "https://github.com/licota/refer",
        "dest": "data/refcocog",
        "instructions": "Download Refer API datasets into 'data/refcocog/'.",
    },
    "custom": {
        "name": "Custom Multimodal Domain Dataset (~1–2 GB)",
        "dest": "data/custom_multimodal",
        "instructions": (
            "1. Create folder 'data/custom_multimodal/images/' and add your image files.\n"
            "2. Place 'annotations.json' inside 'data/custom_multimodal/' with image-caption mapping."
        ),
    },
}


def download_single_dataset(dataset_key: str) -> None:
    key = dataset_key.lower().strip()
    if key not in DATASET_SOURCES:
        print(f"[ERROR] Dataset '{dataset_key}' is not supported.")
        return

    info = DATASET_SOURCES[key]
    dest = info["dest"]
    os.makedirs(dest, exist_ok=True)

    print(f"\n==================================================")
    print(f"Dataset: {info['name']}")
    print(f"Target Directory: {os.path.abspath(dest)}")
    print(f"==================================================")

    if key == "mscoco":
        try:
            print("Downloading MS COCO train2014 images...")
            z1 = download_with_resume(info["train_img_url"], os.path.join(dest, "train2014.zip"))
            extract_archive(z1, dest)

            print("Downloading Karpathy caption splits...")
            z2 = download_with_resume(info["karpathy_url"], os.path.join(dest, "caption_datasets.zip"))
            extract_archive(z2, dest)
        except Exception as e:
            print(f"[DOWNLOAD NOTICE] {e}")
            print(info["instructions"])

    elif key == "vqav2":
        try:
            print("Downloading VQA v2 questions...")
            z1 = download_with_resume(info["train_q_url"], os.path.join(dest, "v2_questions.zip"))
            extract_archive(z1, dest)

            print("Downloading VQA v2 annotations...")
            z2 = download_with_resume(info["train_ann_url"], os.path.join(dest, "v2_annotations.zip"))
            extract_archive(z2, dest)
        except Exception as e:
            print(f"[DOWNLOAD NOTICE] {e}")
            print(info["instructions"])

    else:
        print(f"Instructions:\n{info['instructions']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Multimodal Benchmark Dataset Downloader")
    parser.add_argument(
        "--dataset",
        type=str,
        default="flickr8k",
        choices=list(DATASET_SOURCES.keys()) + ["all"],
        help="Target dataset to setup/download",
    )
    parser.add_argument("--all", action="store_true", help="Download/setup all benchmark datasets")
    args = parser.parse_args()

    if args.all or args.dataset == "all":
        for ds in DATASET_SOURCES:
            download_single_dataset(ds)
    else:
        download_single_dataset(args.dataset)

    print("\n[VERIFICATION STATUS REPORT]")
    res = inspect_dataset_status("data")
    print_status_table(res)


if __name__ == "__main__":
    main()
