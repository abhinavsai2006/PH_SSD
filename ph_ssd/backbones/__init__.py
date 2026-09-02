"""
PH-SSD Backbones Package.
"""

from ph_ssd.backbones.ssd_wrapper import SSDBlock
from ph_ssd.backbones.multimodal_ph_ssd import MultimodalPHSSDBackbone

__all__ = [
    "SSDBlock",
    "MultimodalPHSSDBackbone",
]
