"""
FVD (Fréchet Video Distance) calculator.

Computes video-level distributional distance using I3D features.

Implementation follows:
- Uses I3D pretrained on Kinetics-400
- Fixed clip length (must match between gen and GT)
- Matches pytorch-fvd conventions
"""

from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn

from .base import (
    MetricCalculator,
    MetricResult,
    FeatureAccumulator,
    compute_frechet_distance,
    compute_statistics,
)


class I3DFeatures(nn.Module):
    """
    I3D feature extractor for FVD computation.

    Uses I3D pretrained on Kinetics-400 to extract video features.
    Returns 400-dimensional features from the final layer before logits.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self._model = None  # Lazy loading

    def _load_model(self):
        """Lazily load I3D model."""
        if self._model is not None:
            return

        try:
            # Try to use pytorch-fvd's I3D
            from pytorch_fvd import load_i3d_pretrained
            self._model = load_i3d_pretrained(device=self.device)
        except ImportError:
            # Fallback: try torchvision's video models
            try:
                from torchvision.models.video import r3d_18, R3D_18_Weights
                model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
                # Remove final classification layer
                self._model = nn.Sequential(*list(model.children())[:-1])
                self._model.to(self.device)
                self._model.eval()
                self._feature_dim = 512  # R3D-18 feature dim
            except ImportError:
                raise ImportError(
                    "Please install pytorch-fvd or torchvision with video support. "
                    "pip install pytorch-fvd or pip install torchvision>=0.12"
                )

        for param in self._model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video clips.

        Args:
            x: (N, T, C, H, W) video tensor, values in [0, 1]

        Returns:
            (N, feature_dim) feature tensor
        """
        self._load_model()

        # I3D expects (N, C, T, H, W) format
        x = x.permute(0, 2, 1, 3, 4)

        # Resize if needed (I3D expects 224x224)
        if x.shape[3] != 224 or x.shape[4] != 224:
            N, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x = x.reshape(N, T, C, 224, 224).permute(0, 2, 1, 3, 4)

        # Normalize to [-1, 1]
        x = x * 2 - 1

        with torch.no_grad():
            features = self._model(x.to(self.device))

        # Flatten if needed
        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        return features


class FVDCalculator(MetricCalculator):
    """
    Video-level FVD calculator.

    Computes FVD between generated and GT video clips using I3D features.

    Args:
        device: Computation device
        batch_size: Batch size for feature extraction
        n_frames: Expected number of frames per clip (must be consistent)
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 16,
        n_frames: int = 14,
    ):
        self.batch_size = batch_size
        self.n_frames = n_frames

        super().__init__(device)

        # Initialize feature extractor
        self.feature_extractor = I3DFeatures(device)

    @property
    def name(self) -> str:
        return "FVD"

    def reset(self) -> None:
        """Reset accumulators."""
        self.gen_features = FeatureAccumulator()
        self.gt_features = FeatureAccumulator()

    def _validate_clip_length(self, videos: torch.Tensor) -> None:
        """Validate clip has expected number of frames."""
        if videos.shape[1] != self.n_frames:
            raise ValueError(
                f"Expected {self.n_frames} frames, got {videos.shape[1]}. "
                "FVD requires consistent clip lengths."
            )

    def _extract_features(self, videos: torch.Tensor) -> np.ndarray:
        """
        Extract I3D features from video clips.

        Args:
            videos: (N, T, C, H, W) tensor, values in [0, 1]

        Returns:
            (N, feature_dim) feature array
        """
        videos = videos.to(self.device)

        features = []
        with torch.no_grad():
            for i in range(0, len(videos), self.batch_size):
                batch = videos[i:i + self.batch_size]
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
            gen_frames: (N, T, C, H, W) generated video clips
            gt_frames: (N, T, C, H, W) ground truth video clips
        """
        if gen_frames is not None:
            self._validate_clip_length(gen_frames)
            features = self._extract_features(gen_frames)
            self.gen_features.add(features)

        if gt_frames is not None:
            self._validate_clip_length(gt_frames)
            features = self._extract_features(gt_frames)
            self.gt_features.add(features)

    def compute(self) -> MetricResult:
        """
        Compute FVD from accumulated features.

        Returns:
            MetricResult with FVD score
        """
        if len(self.gen_features) == 0 or len(self.gt_features) == 0:
            raise ValueError("Must accumulate both generated and GT features before computing FVD")

        gen_feats = self.gen_features.get_all()
        gt_feats = self.gt_features.get_all()

        # Compute statistics
        mu_gen, sigma_gen = compute_statistics(gen_feats)
        mu_gt, sigma_gt = compute_statistics(gt_feats)

        # Compute FVD (same formula as FID, just different features)
        fvd = compute_frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)

        return MetricResult(
            name=self.name,
            value=fvd,
            lower_is_better=True,
            details={
                "mu_gen_norm": float(np.linalg.norm(mu_gen)),
                "mu_gt_norm": float(np.linalg.norm(mu_gt)),
            },
            metadata={
                "n_gen_clips": len(self.gen_features),
                "n_gt_clips": len(self.gt_features),
                "n_frames": self.n_frames,
                "feature_dim": gen_feats.shape[1],
            },
        )
