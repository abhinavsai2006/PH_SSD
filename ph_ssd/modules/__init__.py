"""
PH-SSD Core Modules Package.
Includes SD-NPF (Pre-Filter) and VCM-SSD (Variational Boundary Coupler).
"""

from ph_ssd.modules.sd_npf import SymplecticDissipativeNeuralPreFilter
from ph_ssd.modules.vcm_ssd import VariationalCrossModalSSDCoupler

__all__ = [
    "SymplecticDissipativeNeuralPreFilter",
    "VariationalCrossModalSSDCoupler",
]
