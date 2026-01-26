# VersaFace Evaluation - Dataset utilities

from .manifest import EvalSample, EvalManifest
from .loader import EvalDataset
from .splits import create_identity_split, load_identity_split

__all__ = [
    "EvalSample",
    "EvalManifest",
    "EvalDataset",
    "create_identity_split",
    "load_identity_split",
]
