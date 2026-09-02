"""
Synthetic Multimodal Sequence Dataset Loader.
Author: Senior AI Systems Architect
License: Apache 2.0
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict


class SyntheticMultimodalDataset(Dataset):
    """
    Synthetic Multimodal Sequence Dataset for Unit Testing & Pipeline Tracing.
    """

    def __init__(
        self,
        num_samples: int = 100,
        seq_len: int = 64,
        dim_A: int = 768,
        dim_B: int = 768,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_samples: int = num_samples
        self.seq_len: int = seq_len
        self.dim_A: int = dim_A
        self.dim_B: int = dim_B

        generator = torch.Generator().manual_seed(42)
        self.data_A = torch.randn(num_samples, seq_len, dim_A, generator=generator)
        self.data_B = torch.randn(num_samples, seq_len, dim_B, generator=generator)

        for i in range(num_samples):
            # Correlate modality A and modality B signals
            self.data_B[i] = 0.8 * self.data_A[i] + 0.2 * torch.randn(seq_len, dim_B, generator=generator) * noise_std

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "raw_A": self.data_A[idx],
            "raw_B": self.data_B[idx],
            "image_id": f"synthetic_{idx}",
        }
