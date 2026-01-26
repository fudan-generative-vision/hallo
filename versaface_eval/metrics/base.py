"""
Base classes for metric calculators.

All metrics follow the same interface:
1. Accumulate features from generated and ground truth samples
2. Compute final metric after all samples processed
3. Return standardized MetricResult
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np


@dataclass
class MetricResult:
    """
    Standardized metric result container.

    Attributes:
        name: Metric name (e.g., "FID", "FVD", "Sync-C")
        value: Primary metric value
        lower_is_better: Whether lower values indicate better performance
        details: Additional metric details (per-sample scores, etc.)
        metadata: Computation metadata (n_samples, runtime, etc.)
    """
    name: str
    value: float
    lower_is_better: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "lower_is_better": self.lower_is_better,
            "details": self.details,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        direction = "↓" if self.lower_is_better else "↑"
        return f"{self.name}: {self.value:.4f} {direction}"


class MetricCalculator(ABC):
    """
    Abstract base class for metric calculators.

    Usage:
        calculator = FIDCalculator()

        for batch in dataloader:
            calculator.update(gen_frames=batch["gen"], gt_frames=batch["gt"])

        result = calculator.compute()
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated state."""
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update with a batch of samples.

        Args vary by metric type but typically include:
            gen_frames: Generated frames tensor
            gt_frames: Ground truth frames tensor
            gen_audio: Generated audio (for sync metrics)
            gt_audio: Ground truth audio
        """
        pass

    @abstractmethod
    def compute(self) -> MetricResult:
        """Compute final metric from accumulated features."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name for reporting."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"


class FeatureAccumulator:
    """
    Utility class for accumulating features across batches.

    Used by FID, FVD, E-FID to collect features before computing
    distributional distance.
    """

    def __init__(self):
        self.features: List[np.ndarray] = []
        self.n_samples: int = 0

    def add(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        """Add a batch of features."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        if features.ndim == 1:
            features = features[np.newaxis, :]

        self.features.append(features)
        self.n_samples += features.shape[0]

    def get_all(self) -> np.ndarray:
        """Get all accumulated features as single array."""
        if not self.features:
            raise ValueError("No features accumulated")
        return np.concatenate(self.features, axis=0)

    def reset(self) -> None:
        """Clear accumulated features."""
        self.features = []
        self.n_samples = 0

    def __len__(self) -> int:
        return self.n_samples


def compute_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Fréchet Distance between two Gaussians.

    The Fréchet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is:
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

    Args:
        mu1: Mean of first Gaussian
        sigma1: Covariance of first Gaussian
        mu2: Mean of second Gaussian
        sigma2: Covariance of second Gaussian
        eps: Small constant for numerical stability

    Returns:
        Fréchet distance
    """
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, f"Mean shapes differ: {mu1.shape} vs {mu2.shape}"
    assert sigma1.shape == sigma2.shape, f"Covariance shapes differ: {sigma1.shape} vs {sigma2.shape}"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_statistics(features: np.ndarray) -> tuple:
    """
    Compute mean and covariance of features.

    Args:
        features: (N, D) array of features

    Returns:
        (mean, covariance) tuple
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma
