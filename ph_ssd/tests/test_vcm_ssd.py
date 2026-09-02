"""
PyTest Unit Tests for VCM-SSD Module.
"""

import unittest
import torch
from ph_ssd.modules.vcm_ssd import VariationalCrossModalSSDCoupler


class TestVCMSSD(unittest.TestCase):

    def setUp(self) -> None:
        self.d_state = 64
        self.z_dim = 16
        self.batch_size = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.coupler = VariationalCrossModalSSDCoupler(
            d_state=self.d_state, z_dim=self.z_dim
        ).to(self.device)

    def test_shapes(self) -> None:
        h_A = torch.randn(self.batch_size, self.d_state, device=self.device)
        h_B = torch.randn(self.batch_size, self.d_state, device=self.device)
        h_A_next, h_B_next, kl_loss = self.coupler(h_A, h_B)

        self.assertEqual(h_A_next.shape, (self.batch_size, self.d_state))
        self.assertEqual(h_B_next.shape, (self.batch_size, self.d_state))
        self.assertEqual(kl_loss.shape, ())

    def test_kl_divergence_non_negative(self) -> None:
        h_A = torch.randn(self.batch_size, self.d_state, device=self.device)
        h_B = torch.randn(self.batch_size, self.d_state, device=self.device)
        _, _, kl_loss = self.coupler(h_A, h_B)
        self.assertGreaterEqual(kl_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
