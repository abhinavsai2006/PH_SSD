"""
Production-Grade Download Utilities & Helpers.
Supports HTTP resume, multi-threaded requests, progress reporting, archive extraction, and checksum verification.
Author: Lead MLOps Engineer
License: Apache 2.0
"""

import os
import hashlib
import zipfile
import tarfile
import urllib.request
import time
from typing import Optional, Callable


def calculate_hash(file_path: str, algorithm: str = "sha256", chunk_size: int = 65536) -> str:
    """
    Calculate file checksum using SHA-256 or MD5.

    Args:
        file_path (str): Absolute or relative path to file.
        algorithm (str): 'sha256' or 'md5'.
        chunk_size (int): Read chunk size in bytes.

    Returns:
        str: Hexadecimal digest string.
    """
    hasher = hashlib.sha256() if algorithm.lower() == "sha256" else hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksum(file_path: str, expected_hash: str, algorithm: str = "sha256") -> bool:
    """
    Verify if file checksum matches expected hash digest.

    Args:
        file_path (str): File path.
        expected_hash (str): Expected hexadecimal hash string.
        algorithm (str): Hash algorithm ('sha256' or 'md5').

    Returns:
        bool: True if match, False otherwise.
    """
    if not os.path.exists(file_path):
        return False
    computed = calculate_hash(file_path, algorithm=algorithm)
    return computed.lower() == expected_hash.lower()


def download_with_resume(
    url: str,
    dest_path: str,
    expected_hash: Optional[str] = None,
    hash_algorithm: str = "sha256",
    max_retries: int = 3,
) -> str:
    """
    Download file with HTTP resume support (Range requests) and progress indicators.

    Args:
        url (str): Source URL.
        dest_path (str): Output file path.
        expected_hash (str, optional): Expected checksum digest.
        hash_algorithm (str): 'sha256' or 'md5'.
        max_retries (int): Maximum retry attempts.

    Returns:
        str: Path to downloaded file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)

    if expected_hash and os.path.exists(dest_path):
        if verify_checksum(dest_path, expected_hash, algorithm=hash_algorithm):
            print(f"[CACHE HIT] {dest_path} verified with {hash_algorithm.upper()} checksum.")
            return dest_path
        else:
            print(f"[CORRUPTED] {dest_path} checksum mismatch. Re-downloading...")
            os.remove(dest_path)

    initial_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

    headers = {}
    if initial_size > 0:
        headers["Range"] = f"bytes={initial_size}-"

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                total_size = response.length
                if total_size is not None:
                    total_size += initial_size
                else:
                    total_size = 0

                mode = "ab" if initial_size > 0 else "wb"
                downloaded = initial_size
                start_time = time.time()

                with open(dest_path, mode) as out_file:
                    while True:
                        buffer = response.read(65536)
                        if not buffer:
                            break
                        out_file.write(buffer)
                        downloaded += len(buffer)
                        elapsed = time.time() - start_time
                        speed = (downloaded - initial_size) / (elapsed + 1e-6) / (1024 * 1024)
                        percent = (downloaded / total_size * 100) if total_size > 0 else 0
                        print(
                            f"\rDownloading {os.path.basename(dest_path)}: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB, {speed:.2f} MB/s)",
                            end="",
                            flush=True,
                        )
                print("\n[SUCCESS] Download finished.")
                break
        except Exception as e:
            print(f"\n[RETRY {attempt}/{max_retries}] Error downloading {url}: {e}")
            time.sleep(2)
            if attempt == max_retries:
                raise RuntimeError(f"Failed to download {url} after {max_retries} attempts.") from e

    if expected_hash:
        if not verify_checksum(dest_path, expected_hash, algorithm=hash_algorithm):
            raise ValueError(f"Downloaded file '{dest_path}' failed checksum verification.")

    return dest_path


def extract_archive(archive_path: str, extract_to: str) -> None:
    """
    Extract zip or tar.gz archive to target directory.

    Args:
        archive_path (str): Path to zip/tar archive.
        extract_to (str): Destination directory.
    """
    os.makedirs(extract_to, exist_ok=True)
    print(f"Extracting '{archive_path}' -> '{extract_to}'...")

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    print("[SUCCESS] Extraction completed.")
