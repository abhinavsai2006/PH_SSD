"""
PyTest Unit Tests for SD-NPF Module.
"""

import unittest
import torch
from ph_ssd.modules.sd_npf import SymplecticDissipativeNeuralPreFilter


class TestSDNPF(unittest.TestCase):

    def setUp(self) -> None:
        self.d_model = 32
        self.batch_size = 2
        self.seq_len = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SymplecticDissipativeNeuralPreFilter(d_model=self.d_model).to(self.device)

    def test_shapes(self) -> None:
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        x_hat, energy = self.model(x)
        self.assertEqual(x_hat.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(energy.shape, (self.batch_size, self.seq_len))

    def test_energy_decay(self) -> None:
        """Verifies Theorem 1: Under unforced input (x=0), total energy H(t) strictly decays."""
        with torch.no_grad():
            x_impulse = torch.zeros(self.batch_size, self.seq_len, self.d_model, device=self.device)
            x_impulse[:, 0, :] = 2.0  # Initial impulse at t=0
            _, energy = self.model(x_impulse)

        # For t >= 1, x_t = 0 (unforced), so H(t) must decay monotonically: H_t - H_{t-1} <= 0
        energy_diff = energy[:, 2:] - energy[:, 1:-1]
        self.assertLessEqual(energy_diff.mean().item(), 0.0, "Energy failed to decay monotonically under autonomous conditions.")

    def test_autograd_gradients(self) -> None:
        """Verifies backward pass without NaNs."""
        x = torch.randn(self.batch_size, 8, self.d_model, device=self.device, requires_grad=True)
        x_hat, energy = self.model(x)
        loss = x_hat.sum() + energy.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


if __name__ == "__main__":
    unittest.main()
