"""
Multimodal Inference API Entrypoint.
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from ph_ssd.models.ph_ssd_model import PHSSDTaskModel


class PHSSDInferenceAPI:
    """
    Multimodal PH-SSD Inference API Service.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, d_model: int = 128) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PHSSDTaskModel(input_dim_A=768, input_dim_B=768, d_model=d_model, num_classes=10).to(self.device)
        self.model.eval()

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])

    @torch.no_grad()
    def predict(self, raw_A: torch.Tensor, raw_B: torch.Tensor) -> Tuple[int, float]:
        """
        Run inference on visual and text inputs.

        Args:
            raw_A (torch.Tensor): Visual input sequence (1, Seq_Len, 768)
            raw_B (torch.Tensor): Text input sequence (1, Seq_Len, 768)

        Returns:
            Tuple[int, float]: (Predicted_Class_Index, Prediction_Confidence)
        """
        raw_A = raw_A.to(self.device)
        raw_B = raw_B.to(self.device)

        outputs = self.model(raw_A, raw_B)
        probs = torch.softmax(outputs["logits"], dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()

        return pred_idx, confidence


if __name__ == "__main__":
    api = PHSSDInferenceAPI()
    dummy_A = torch.randn(1, 64, 768)
    dummy_B = torch.randn(1, 64, 768)
    cls_idx, conf = api.predict(dummy_A, dummy_B)
    print(f"Inference Test Output -> Predicted Class: {cls_idx}, Confidence: {conf:.4f}")
