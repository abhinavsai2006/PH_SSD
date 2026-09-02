"""
Structured Metric Logger Utility with Immediate Console Flush.
Author: Senior AI Systems Architect
License: Apache 2.0
"""

import sys
import logging
from typing import Dict, Any


class MetricLogger:
    """
    Structured Metric Logger for Console and File Logging.
    """

    def __init__(self, name: str = "PH-SSD") -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log dictionary of metrics for an epoch.

        Args:
            epoch (int): Epoch index.
            metrics (Dict[str, float]): Dictionary of metric key-values.
        """
        formatted = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch [{epoch:03d}] -> {formatted}")
        sys.stdout.flush()
