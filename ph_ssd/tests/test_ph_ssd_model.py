"""
PyTest End-to-End Pipeline & Training Integration Test for Contrastive Retrieval.
"""

import unittest
import torch
from torch.utils.data import DataLoader

from ph_ssd.datasets.synthetic_multimodal import SyntheticMultimodalDataset
from ph_ssd.models.ph_ssd_model import PHSSDTaskModel
from ph_ssd.losses.ph_ssd_loss import PHSSDLoss
from ph_ssd.training.trainer import PHSSDTrainer
from ph_ssd.evaluation.evaluator import PHSSDEvaluator


class TestPHSSDModelPipeline(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = SyntheticMultimodalDataset(num_samples=20, seq_len=16, dim_A=16, dim_B=16)
        self.dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)

        self.model = PHSSDTaskModel(input_dim_A=16, input_dim_B=16, d_model=32, d_embed=32, d_state=16, z_dim=8, n_layers=2)
        self.criterion = PHSSDLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.trainer = PHSSDTrainer(self.model, self.criterion, self.optimizer, device=self.device, use_amp=False)
        self.evaluator = PHSSDEvaluator(self.model, device=self.device)

    def test_single_training_epoch(self) -> None:
        train_metrics = self.trainer.train_epoch(self.dataloader)
        self.assertIn("epoch/loss", train_metrics)
        self.assertIn("epoch/contrastive_loss", train_metrics)
        self.assertGreaterEqual(train_metrics["epoch/loss"], 0.0)

    def test_evaluation(self) -> None:
        eval_metrics = self.evaluator.evaluate(self.dataloader)
        self.assertIn("retrieval/i2t_r1", eval_metrics)
        self.assertIn("retrieval/t2i_r1", eval_metrics)
        self.assertGreaterEqual(eval_metrics["retrieval/i2t_r1"], 0.0)


if __name__ == "__main__":
    unittest.main()
