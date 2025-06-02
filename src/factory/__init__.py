"""Factory - A Python package for crystallographic data processing.

This package provides the pipeline from shoeboxes to structure factors.
"""

__version__ = "0.1.0"

from factory import CC12, anomalous_peaks
from factory.model import Model

__all__ = [
    "Model",
    "CC12",
    "anomalous_peaks",
]
