"""
ONNX Model Export Utility.
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
import torch
from ph_ssd.models.ph_ssd_model import PHSSDTaskModel


def export_to_onnx(output_path: str = "checkpoints/ph_ssd.onnx") -> str:
    """Export trained PH-SSD model to ONNX format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = PHSSDTaskModel(input_dim_A=64, input_dim_B=64, d_model=64, num_classes=5).eval()

    dummy_A = torch.randn(1, 32, 64)
    dummy_B = torch.randn(1, 32, 64)

    torch.onnx.export(
        model,
        (dummy_A, dummy_B),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["raw_A", "raw_B"],
        output_names=["logits"],
        dynamic_axes={"raw_A": {0: "batch_size", 1: "seq_len"}, "raw_B": {0: "batch_size", 1: "seq_len"}},
    )
    print(f"Model exported successfully to ONNX: {output_path}")
    return output_path


if __name__ == "__main__":
    export_to_onnx()
