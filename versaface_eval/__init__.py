# VersaFace Evaluation Framework
# Implements Hallo-style evaluation protocol with extensions for motion/expression alignment

__version__ = "0.1.0"

from .datasets.manifest import EvalSample, EvalManifest
from .datasets.loader import EvalDataset
from .metrics import FIDCalculator, FVDCalculator, SyncNetCalculator, EFIDCalculator

__all__ = [
    "EvalSample",
    "EvalManifest",
    "EvalDataset",
    "FIDCalculator",
    "FVDCalculator",
    "SyncNetCalculator",
    "EFIDCalculator",
]
