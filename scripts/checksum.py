"""
Checksum Verification Tool for Multimodal Benchmark Archives and Datasets.
Author: Lead MLOps Engineer
License: Apache 2.0
"""

import sys
import argparse
from scripts.download_utils import calculate_hash, verify_checksum


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Dataset File SHA-256 / MD5 Checksums")
    parser.add_argument("--file", type=str, required=True, help="Path to target file")
    parser.add_argument("--hash", type=str, required=True, help="Expected hexadecimal hash digest")
    parser.add_argument("--algo", type=str, default="sha256", choices=["sha256", "md5"], help="Hash algorithm")
    args = parser.parse_args()

    print(f"Computing {args.algo.upper()} checksum for '{args.file}'...")
    actual = calculate_hash(args.file, algorithm=args.algo)

    print(f"Computed: {actual}")
    print(f"Expected: {args.hash}")

    if actual.lower() == args.hash.lower():
        print("[MATCH] Checksum verified successfully!")
        sys.exit(0)
    else:
        print("[MISMATCH] Checksum verification failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
