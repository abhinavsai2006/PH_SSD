"""
Dataset Integrity & JSON Schema Verification Script.
Author: Lead MLOps Engineer
License: Apache 2.0
"""

import sys
import argparse
from scripts.dataset_status import inspect_dataset_status, print_status_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Multimodal Benchmark Datasets")
    parser.add_argument("--dataset", type=str, default="all", help="Target dataset name or 'all'")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data folder")
    args = parser.parse_args()

    results = inspect_dataset_status(args.data_dir)

    if args.dataset != "all":
        target = args.dataset.lower().strip()
        if target in results:
            results = {target: results[target]}
        else:
            print(f"[ERROR] Dataset '{args.dataset}' is not supported.")
            sys.exit(1)

    print_status_table(results)


if __name__ == "__main__":
    main()
