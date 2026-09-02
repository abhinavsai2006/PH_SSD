"""
Dataset Archive Extraction Tool.
Author: Lead MLOps Engineer
License: Apache 2.0
"""

import os
import argparse
from scripts.download_utils import extract_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Dataset Zip/Tar Archives")
    parser.add_argument("--archive", type=str, help="Path to specific zip or tar archive file")
    parser.add_argument("--dest", type=str, default="data", help="Extraction destination root")
    parser.add_argument("--all", action="store_true", help="Extract all archives in data/ directory")
    args = parser.parse_args()

    if args.all:
        print(f"Scanning '{args.dest}' for zip and tar archives...")
        archives_found = 0
        for root, _, files in os.walk(args.dest):
            for file in files:
                if file.endswith((".zip", ".tar.gz", ".tgz")):
                    archive_path = os.path.join(root, file)
                    extract_archive(archive_path, root)
                    archives_found += 1
        print(f"[COMPLETE] Extracted {archives_found} archive(s).")
    elif args.archive:
        extract_archive(args.archive, args.dest)
    else:
        print("[ERROR] Please specify --archive <path> or --all")


if __name__ == "__main__":
    main()
