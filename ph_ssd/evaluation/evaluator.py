"""
Unified Evaluation Pipeline for PH-SSD Multimodal Retrieval.
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from ph_ssd.models.ph_ssd_model import PHSSDTaskModel
from ph_ssd.evaluation.retrieval_metrics import compute_retrieval_recalls
from ph_ssd.evaluation.efficiency_profiler import profile_model_efficiency


class PHSSDEvaluator:
    """
    Unified PH-SSD Model Evaluator for Cross-Modal Retrieval and Model Efficiency.

    Computes:
      - Cross-Modal Image-to-Text & Text-to-Image Retrieval Recalls (R@1, R@5, R@10, Mean Rank, Median Rank, mR)
      - VCM-SSD KL Loss & Loss Metrics
      - Hardware Efficiency Metrics (Latency, Throughput, Memory, FLOPs, Params)
    """

    def __init__(
        self,
        model: PHSSDTaskModel,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model: PHSSDTaskModel = model
        self.device: torch.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance across DataLoader by extracting normalized embeddings
        and computing exact cross-modal retrieval rankings with multi-caption mapping.

        Args:
            dataloader (DataLoader): Evaluation DataLoader.

        Returns:
            Dict[str, float]: Evaluated metrics dictionary.
        """
        self.model.eval()
        all_embed_A = []
        all_embed_B = []
        total_kl_loss = 0.0
        total_samples = 0

        # Mapping data structures for multi-caption evaluation
        image_id_to_idx: Dict[str, int] = {}
        unique_image_embeds: List[torch.Tensor] = []
        
        img_to_txt_map: Dict[int, List[int]] = {}
        txt_to_img_map: Dict[int, int] = {}

        caption_counter = 0

        for batch in dataloader:
            raw_A = batch["raw_A"].to(self.device)
            raw_B = batch["raw_B"].to(self.device)

            outputs = self.model(raw_A, raw_B)
            
            embed_A = outputs["embed_A"].detach().cpu() # (B, d_embed)
            embed_B = outputs["embed_B"].detach().cpu() # (B, d_embed)
            
            all_embed_B.append(embed_B)
            
            image_ids = batch.get("image_id", [f"img_{i}" for i in range(raw_A.size(0))])
            
            for b in range(raw_A.size(0)):
                img_id = image_ids[b]
                if img_id not in image_id_to_idx:
                    img_idx = len(image_id_to_idx)
                    image_id_to_idx[img_id] = img_idx
                    unique_image_embeds.append(embed_A[b].unsqueeze(0))
                else:
                    img_idx = image_id_to_idx[img_id]

                txt_idx = caption_counter
                img_to_txt_map.setdefault(img_idx, []).append(txt_idx)
                txt_to_img_map[txt_idx] = img_idx
                caption_counter += 1

            total_kl_loss += outputs["kl_loss"].item() * raw_A.size(0)
            total_samples += raw_A.size(0)

        if total_samples == 0:
            return {}

        embed_A_unique = torch.cat(unique_image_embeds, dim=0) # (N_img, d_embed)
        embed_B_cat = torch.cat(all_embed_B, dim=0)            # (N_txt, d_embed)

        # Cross-modal cosine similarity matrix (N_img, N_txt)
        sim_mat = torch.matmul(embed_A_unique, embed_B_cat.t())

        retrieval_metrics = compute_retrieval_recalls(
            sim_mat,
            img_to_txt_map=img_to_txt_map,
            txt_to_img_map=txt_to_img_map
        )

        eval_results = {
            "eval/avg_kl_loss": float(total_kl_loss / total_samples),
            "eval/samples": float(total_samples),
        }
        eval_results.update(retrieval_metrics)

        return eval_results

    def profile_efficiency(self, sample_A: torch.Tensor, sample_B: torch.Tensor) -> Dict[str, Any]:
        """Profile latency, memory, throughput, and FLOPs."""
        return profile_model_efficiency(self.model, sample_A, sample_B)
