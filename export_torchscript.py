"""
TorchScript Export Utility.
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
import torch
from ph_ssd.models.ph_ssd_model import PHSSDTaskModel


def export_to_torchscript(output_path: str = "checkpoints/ph_ssd.pt") -> str:
    """Export PH-SSD model to TorchScript via tracing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = PHSSDTaskModel(input_dim_A=64, input_dim_B=64, d_model=64, num_classes=5).eval()

    dummy_A = torch.randn(1, 32, 64)
    dummy_B = torch.randn(1, 32, 64)

    traced_model = torch.jit.trace(model, (dummy_A, dummy_B), strict=False)
    traced_model.save(output_path)
    print(f"Model exported successfully to TorchScript: {output_path}")
    return output_path


if __name__ == "__main__":
    export_to_torchscript()
