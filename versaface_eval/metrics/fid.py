"""
FID (Fréchet Inception Distance) calculator.

Computes frame-level FID between generated and ground truth video frames.

Definition:
    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2))

Where:
    - μ_r, Σ_r = mean and covariance of real image features
    - μ_g, Σ_g = mean and covariance of generated image features
    - Features extracted from InceptionV3 pool3 layer (2048-dim)

Direction: Lower is better (0 = identical distributions)

Implementation follows:
- Uses InceptionV3 features (2048-dim from pool3 layer)
- Frame sampling policy: uniform or all frames
- Matches clean-fid conventions for reproducibility

Sources:
- FID Wikipedia: https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance
- PyTorch-Metrics FID: https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
- Original Paper: "GANs Trained by a Two Time-Scale Update Rule..."
"""

from typing import Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d

from .base import (
    MetricCalculator,
    MetricResult,
    FeatureAccumulator,
    compute_frechet_distance,
    compute_statistics,
)


class InceptionV3Features(nn.Module):
    """
    InceptionV3 feature extractor for FID computation.

    Extracts 2048-dimensional features from the pool3 layer.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

        # Load pretrained InceptionV3
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        except ImportError:
            # Fallback for older torchvision
            from torchvision.models import inception_v3
            inception = inception_v3(pretrained=True)

        # Remove final layers - we want features not predictions
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )

        self.to(device)
        self.eval()

        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            x: (N, 3, H, W) tensor of images, normalized to [-1, 1]

        Returns:
            (N, 2048) feature tensor
        """
        # Inception expects 299x299 input
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = nn.functional.interpolate(
                x, size=(299, 299), mode='bilinear', align_corners=False
            )

        # Forward through blocks
        x = self.blocks(x)

        # Adaptive average pooling to get 2048-dim features
        x = adaptive_avg_pool2d(x, output_size=(1, 1))
        x = x.view(x.size(0), -1)

        return x


class FIDCalculator(MetricCalculator):
    """
    Frame-level FID calculator.

    Computes FID between frames sampled from generated and GT videos.

    Args:
        device: Computation device
        batch_size: Batch size for feature extraction
        sample_mode: "all" to use all frames, "uniform" to sample uniformly
        n_samples_per_clip: Number of frames to sample per clip (if uniform)
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 64,
        sample_mode: str = "uniform",
        n_samples_per_clip: int = 8,
    ):
        self.batch_size = batch_size
        self.sample_mode = sample_mode
        self.n_samples_per_clip = n_samples_per_clip

        super().__init__(device)

        # Initialize feature extractor
        self.feature_extractor = InceptionV3Features(device)

    @property
    def name(self) -> str:
        return "FID"

    def reset(self) -> None:
        """Reset accumulators."""
        self.gen_features = FeatureAccumulator()
        self.gt_features = FeatureAccumulator()

    def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Sample frames according to sampling policy.

        Args:
            frames: (N, T, C, H, W) video tensor

        Returns:
            (M, C, H, W) sampled frames
        """
        N, T, C, H, W = frames.shape

        if self.sample_mode == "all":
            # Return all frames flattened
            return frames.view(N * T, C, H, W)

        elif self.sample_mode == "uniform":
            # Uniformly sample n_samples_per_clip frames from each clip
            indices = np.linspace(0, T - 1, self.n_samples_per_clip, dtype=int)
            sampled = frames[:, indices]  # (N, n_samples, C, H, W)
            return sampled.view(N * self.n_samples_per_clip, C, H, W)

        else:
            raise ValueError(f"Unknown sample_mode: {self.sample_mode}")

    def _extract_features(self, frames: torch.Tensor) -> np.ndarray:
        """
        Extract InceptionV3 features from frames.

        Args:
            frames: (N, C, H, W) tensor, values in [0, 1]

        Returns:
            (N, 2048) feature array
        """
        # Normalize to [-1, 1] as expected by Inception
        frames = frames * 2 - 1
        frames = frames.to(self.device)

        features = []
        with torch.no_grad():
            for i in range(0, len(frames), self.batch_size):
                batch = frames[i:i + self.batch_size]
                feat = self.feature_extractor(batch)
                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def update(
        self,
        gen_frames: Optional[torch.Tensor] = None,
        gt_frames: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """
        Update with a batch of video clips.

        Args:
            gen_frames: (N, T, C, H, W) generated video frames
            gt_frames: (N, T, C, H, W) ground truth video frames
        """
        if gen_frames is not None:
            sampled = self._sample_frames(gen_frames)
            features = self._extract_features(sampled)
            self.gen_features.add(features)

        if gt_frames is not None:
            sampled = self._sample_frames(gt_frames)
            features = self._extract_features(sampled)
            self.gt_features.add(features)

    def update_from_paths(
        self,
        gen_frame_paths: Optional[List[str]] = None,
        gt_frame_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Update from image file paths (for large-scale evaluation).

        Args:
            gen_frame_paths: List of paths to generated frames
            gt_frame_paths: List of paths to GT frames
        """
        from PIL import Image
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
        ])

        def load_and_extract(paths: List[str]) -> np.ndarray:
            frames = []
            for path in paths:
                img = Image.open(path).convert('RGB')
                frames.append(transform(img))

            frames = torch.stack(frames)
            return self._extract_features(frames)

        if gen_frame_paths:
            features = load_and_extract(gen_frame_paths)
            self.gen_features.add(features)

        if gt_frame_paths:
            features = load_and_extract(gt_frame_paths)
            self.gt_features.add(features)

    def compute(self) -> MetricResult:
        """
        Compute FID from accumulated features.

        Returns:
            MetricResult with FID score
        """
        if len(self.gen_features) == 0 or len(self.gt_features) == 0:
            raise ValueError("Must accumulate both generated and GT features before computing FID")

        gen_feats = self.gen_features.get_all()
        gt_feats = self.gt_features.get_all()

        # Compute statistics
        mu_gen, sigma_gen = compute_statistics(gen_feats)
        mu_gt, sigma_gt = compute_statistics(gt_feats)

        # Compute FID
        fid = compute_frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)

        return MetricResult(
            name=self.name,
            value=fid,
            lower_is_better=True,
            details={
                "mu_gen_norm": float(np.linalg.norm(mu_gen)),
                "mu_gt_norm": float(np.linalg.norm(mu_gt)),
            },
            metadata={
                "n_gen_samples": len(self.gen_features),
                "n_gt_samples": len(self.gt_features),
                "sample_mode": self.sample_mode,
                "n_samples_per_clip": self.n_samples_per_clip,
                "feature_dim": gen_feats.shape[1],
            },
        )


def compute_fid_from_directories(
    gen_dir: str,
    gt_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
) -> float:
    """
    Convenience function to compute FID between two directories of images.

    Args:
        gen_dir: Directory containing generated images
        gt_dir: Directory containing ground truth images
        device: Computation device
        batch_size: Batch size for feature extraction

    Returns:
        FID score
    """
    from pathlib import Path

    gen_paths = sorted(Path(gen_dir).glob("*.png")) + sorted(Path(gen_dir).glob("*.jpg"))
    gt_paths = sorted(Path(gt_dir).glob("*.png")) + sorted(Path(gt_dir).glob("*.jpg"))

    calculator = FIDCalculator(device=device, batch_size=batch_size)
    calculator.update_from_paths(
        gen_frame_paths=[str(p) for p in gen_paths],
        gt_frame_paths=[str(p) for p in gt_paths],
    )

    result = calculator.compute()
    return result.value
