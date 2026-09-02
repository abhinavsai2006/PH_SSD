"""
Unit tests for Image-Text Retrieval Recalls and Contrastive Loss.
Author: Reproducibility Auditor
License: Apache 2.0
"""

import unittest
import torch
import numpy as np

from ph_ssd.evaluation.retrieval_metrics import compute_retrieval_recalls
from ph_ssd.losses.ph_ssd_loss import PHSSDLoss
from ph_ssd.models.ph_ssd_model import PHSSDTaskModel


class TestRetrievalMetrics(unittest.TestCase):

    def test_perfect_retrieval(self):
        # Identity matrix: perfect 1-to-1 diagonal matching
        sim_matrix = torch.eye(10)
        metrics = compute_retrieval_recalls(sim_matrix)

        self.assertEqual(metrics["retrieval/i2t_r1"], 100.0)
        self.assertEqual(metrics["retrieval/i2t_r5"], 100.0)
        self.assertEqual(metrics["retrieval/i2t_r10"], 100.0)
        self.assertEqual(metrics["retrieval/t2i_r1"], 100.0)
        self.assertEqual(metrics["retrieval/t2i_r5"], 100.0)
        self.assertEqual(metrics["retrieval/t2i_r10"], 100.0)
        self.assertEqual(metrics["retrieval/i2t_mean_rank"], 1.0)
        self.assertEqual(metrics["retrieval/t2i_mean_rank"], 1.0)

    def test_contrastive_loss_shape(self):
        loss_fn = PHSSDLoss(contrastive_weight=1.0, kl_weight=1e-3)
        embed_A = torch.randn(8, 128)
        embed_B = torch.randn(8, 128)
        outputs = {
            "embed_A": torch.nn.functional.normalize(embed_A, p=2, dim=-1),
            "embed_B": torch.nn.functional.normalize(embed_B, p=2, dim=-1),
            "logit_scale": torch.tensor(14.28),
            "kl_loss": torch.tensor(0.005),
        }
        loss, metrics = loss_fn(outputs)

        self.assertIsInstance(loss.item(), float)
        self.assertGreater(metrics["loss/contrastive"], 0.0)
        self.assertGreater(metrics["loss/total"], 0.0)

    def test_model_forward_shape(self):
        model = PHSSDTaskModel(input_dim_A=768, input_dim_B=768, d_model=128, d_embed=128)
        raw_A = torch.randn(2, 64, 768)
        raw_B = torch.randn(2, 64, 768)
        outputs = model(raw_A, raw_B)

        self.assertEqual(outputs["embed_A"].shape, (2, 128))
        self.assertEqual(outputs["embed_B"].shape, (2, 128))
        self.assertTrue(torch.is_tensor(outputs["kl_loss"]))


if __name__ == "__main__":
    unittest.main()
