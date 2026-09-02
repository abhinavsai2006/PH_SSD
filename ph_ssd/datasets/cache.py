"""
Disk Feature Caching Utility for Accelerated Dataset Loading.
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
import torch
from typing import Dict, Any, Optional


class FeatureCacheManager:
    """
    Manages caching extracted image/text feature embeddings on disk.
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir: str = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, key: str) -> str:
        safe_key = "".join([c if c.isalnum() else "_" for c in key])
        return os.path.join(self.cache_dir, f"{safe_key}.pt")

    def exists(self, key: str) -> bool:
        return os.path.exists(self.get_cache_path(key))

    def save(self, key: str, data: Dict[str, torch.Tensor]) -> None:
        filepath = self.get_cache_path(key)
        torch.save(data, filepath)

    def load(self, key: str) -> Dict[str, torch.Tensor]:
        filepath = self.get_cache_path(key)
        return torch.load(filepath, map_location="cpu")
