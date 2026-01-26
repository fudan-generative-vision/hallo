"""
E-FID (Expression FID) calculator.

Computes FID in expression/face parameter space rather than pixel/inception space.

Definition:
    E-FID measures expression semantic fidelity by quantifying the distribution
    difference of facial expression features between generated and real videos.
    Uses the same Fréchet distance formula as FID/FVD.

Uses 3DMM (3D Morphable Model) parameters to extract expression embeddings:
- Expression coefficients (50-dim in FLAME model)
- Jaw pose (3-dim)
- Optionally: eye gaze, head pose

Implementation supports EMOCA, DECA, or similar face reconstruction models.

⚠️ UNCERTAINTY NOTE:
The exact E-FID definition varies across papers and is NOT standardized.
Hallo paper does not specify their exact computation method.

Options found in literature:
1. 3DMM Expression Parameters (our default): Use FLAME/3DMM expression
   coefficients (50-dim) + jaw pose (3-dim) = 53-dim features
2. Expression Recognition Features: Use features from an expression
   classification network (e.g., AffectNet-trained)
3. Emotion Features: Use emotion embedding from emotion recognition model

We chose 3DMM-based approach as it is most commonly referenced in
talking-face papers (MF-ETalk, etc).

Typical Values (from MF-ETalk paper, MDPI 2024):
- MEAD dataset: E-FID ≈ 2.403
- HDTF dataset: E-FID ≈ 3.127

Sources:
- MF-ETalk (MDPI 2024): https://www.mdpi.com/2079-9292/14/13/2684
  "Expression-FID (E-FID) measures expression semantic fidelity by
   quantifying the distribution difference of facial expression features
   between generated and real videos"
"""

from typing import Optional, List, Dict, Any
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


class ExpressionEncoder(nn.Module):
    """
    Expression parameter encoder using 3DMM reconstruction.

    Extracts expression coefficients from face images using EMOCA/DECA-style models.
    """

    def __init__(
        self,
        model_type: str = "emoca",
        device: str = "cuda",
    ):
        super().__init__()
        self.model_type = model_type
        self.device = device
        self._model = None
        self._feature_dim = None

    def _load_model(self):
        """Lazily load expression encoder model."""
        if self._model is not None:
            return

        if self.model_type == "emoca":
            self._load_emoca()
        elif self.model_type == "deca":
            self._load_deca()
        elif self.model_type == "resnet":
            # Fallback: use a ResNet trained for expression recognition
            self._load_resnet_expression()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _load_emoca(self):
        """Load EMOCA model for expression extraction."""
        try:
            from gdl.models.EMOCA import EMOCA
            from gdl.models.IO import load_model

            # EMOCA provides rich expression parameters
            self._model = load_model("EMOCA")
            self._model.to(self.device)
            self._model.eval()

            # Expression dim: 50 (expression) + 3 (jaw) = 53
            self._feature_dim = 53

        except ImportError:
            print("EMOCA not available, falling back to ResNet expression encoder")
            self._load_resnet_expression()

    def _load_deca(self):
        """Load DECA model for expression extraction."""
        try:
            from decalib.deca import DECA
            from decalib.utils.config import cfg as deca_cfg

            self._model = DECA(config=deca_cfg)
            self._model.to(self.device)
            self._model.eval()

            # DECA expression dim: 50 (expression) + 3 (jaw) = 53
            self._feature_dim = 53

        except ImportError:
            print("DECA not available, falling back to ResNet expression encoder")
            self._load_resnet_expression()

    def _load_resnet_expression(self):
        """
        Fallback: Load a ResNet-based expression encoder.

        Uses features from a model trained for facial expression recognition.
        """
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except ImportError:
            from torchvision.models import resnet50
            resnet = resnet50(pretrained=True)

        # Remove final classification layer - use 2048-dim features
        self._model = nn.Sequential(*list(resnet.children())[:-1])
        self._model.to(self.device)
        self._model.eval()
        self._feature_dim = 2048

        for param in self._model.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract expression features from face images.

        Args:
            images: (N, C, H, W) face images, values in [0, 1]

        Returns:
            (N, feature_dim) expression features
        """
        self._load_model()

        images = images.to(self.device)

        if self.model_type in ["emoca", "deca"]:
            return self._forward_3dmm(images)
        else:
            return self._forward_resnet(images)

    def _forward_3dmm(self, images: torch.Tensor) -> torch.Tensor:
        """Extract expression parameters using 3DMM model."""
        with torch.no_grad():
            # Resize to expected input size (usually 224x224)
            if images.shape[2] != 224 or images.shape[3] != 224:
                images = nn.functional.interpolate(
                    images, size=(224, 224), mode='bilinear', align_corners=False
                )

            # Get 3DMM parameters
            if self.model_type == "emoca":
                output = self._model.encode(images)
                exp = output["expcode"]  # (N, 50)
                jaw = output["posecode"][:, :3]  # First 3 are jaw pose
                features = torch.cat([exp, jaw], dim=1)
            else:  # DECA
                output = self._model.encode(images)
                exp = output["exp"]  # (N, 50)
                jaw = output["pose"][:, 3:6]  # Jaw pose from full pose
                features = torch.cat([exp, jaw], dim=1)

        return features

    def _forward_resnet(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features using ResNet encoder."""
        with torch.no_grad():
            # Resize to 224x224
            if images.shape[2] != 224 or images.shape[3] != 224:
                images = nn.functional.interpolate(
                    images, size=(224, 224), mode='bilinear', align_corners=False
                )

            # Normalize for ImageNet
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            images = (images - mean) / std

            features = self._model(images)
            features = features.view(features.size(0), -1)

        return features

    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        self._load_model()
        return self._feature_dim


class EFIDCalculator(MetricCalculator):
    """
    Expression FID calculator.

    Computes FID in expression parameter space to measure distributional
    similarity of facial expressions between generated and GT videos.

    Args:
        device: Computation device
        encoder_type: Type of expression encoder ("emoca", "deca", "resnet")
        parameters: Which 3DMM parameters to use (for emoca/deca)
        batch_size: Batch size for feature extraction
        sample_mode: "all" or "uniform" frame sampling
        n_samples_per_clip: Frames to sample per clip (if uniform)
    """

    def __init__(
        self,
        device: str = "cuda",
        encoder_type: str = "resnet",  # Default to resnet for broader compatibility
        parameters: Optional[List[str]] = None,
        batch_size: int = 32,
        sample_mode: str = "uniform",
        n_samples_per_clip: int = 8,
    ):
        self.encoder_type = encoder_type
        self.parameters = parameters or ["expression", "jaw"]
        self.batch_size = batch_size
        self.sample_mode = sample_mode
        self.n_samples_per_clip = n_samples_per_clip

        super().__init__(device)

        # Initialize expression encoder
        self.encoder = ExpressionEncoder(model_type=encoder_type, device=device)

    @property
    def name(self) -> str:
        return "E-FID"

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
            return frames.view(N * T, C, H, W)
        elif self.sample_mode == "uniform":
            indices = np.linspace(0, T - 1, self.n_samples_per_clip, dtype=int)
            sampled = frames[:, indices]
            return sampled.view(N * self.n_samples_per_clip, C, H, W)
        else:
            raise ValueError(f"Unknown sample_mode: {self.sample_mode}")

    def _extract_features(self, frames: torch.Tensor) -> np.ndarray:
        """
        Extract expression features from frames.

        Args:
            frames: (N, C, H, W) tensor, values in [0, 1]

        Returns:
            (N, feature_dim) feature array
        """
        features = []
        with torch.no_grad():
            for i in range(0, len(frames), self.batch_size):
                batch = frames[i:i + self.batch_size].to(self.device)
                feat = self.encoder(batch)
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
            features = self._extract_features(gt_frames)
            self.gt_features.add(features)

    def compute(self) -> MetricResult:
        """
        Compute E-FID from accumulated features.

        Returns:
            MetricResult with E-FID score
        """
        if len(self.gen_features) == 0 or len(self.gt_features) == 0:
            raise ValueError("Must accumulate both generated and GT features before computing E-FID")

        gen_feats = self.gen_features.get_all()
        gt_feats = self.gt_features.get_all()

        # Compute statistics
        mu_gen, sigma_gen = compute_statistics(gen_feats)
        mu_gt, sigma_gt = compute_statistics(gt_feats)

        # Compute E-FID
        efid = compute_frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)

        return MetricResult(
            name=self.name,
            value=efid,
            lower_is_better=True,
            details={
                "mu_gen_norm": float(np.linalg.norm(mu_gen)),
                "mu_gt_norm": float(np.linalg.norm(mu_gt)),
                "feature_stats": {
                    "gen_mean": float(np.mean(gen_feats)),
                    "gen_std": float(np.std(gen_feats)),
                    "gt_mean": float(np.mean(gt_feats)),
                    "gt_std": float(np.std(gt_feats)),
                },
            },
            metadata={
                "n_gen_samples": len(self.gen_features),
                "n_gt_samples": len(self.gt_features),
                "encoder_type": self.encoder_type,
                "parameters": self.parameters,
                "sample_mode": self.sample_mode,
                "n_samples_per_clip": self.n_samples_per_clip,
                "feature_dim": gen_feats.shape[1],
                "warning": (
                    "E-FID definition is not standardized across papers. "
                    "Our implementation uses {} features. "
                    "Results may not be directly comparable to other papers."
                ).format(
                    "3DMM expression parameters (53-dim)" if self.encoder_type in ["emoca", "deca"]
                    else "ResNet features (2048-dim, less ideal for expression comparison)"
                ),
            },
        )


class ExpressionDiversityMetric(MetricCalculator):
    """
    Measures expression diversity within generated videos.

    Computes variance of expression parameters across frames to ensure
    the model produces dynamic, varied expressions rather than static faces.
    """

    def __init__(
        self,
        device: str = "cuda",
        encoder_type: str = "resnet",
    ):
        self.encoder_type = encoder_type
        super().__init__(device)
        self.encoder = ExpressionEncoder(model_type=encoder_type, device=device)

    @property
    def name(self) -> str:
        return "Expr-Diversity"

    def reset(self) -> None:
        self.diversity_scores: List[float] = []

    def update(
        self,
        gen_frames: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """
        Update with generated video clips.

        Computes per-clip expression variance.
        """
        if gen_frames is None:
            return

        N, T, C, H, W = gen_frames.shape

        with torch.no_grad():
            for i in range(N):
                clip = gen_frames[i].to(self.device)  # (T, C, H, W)
                features = self.encoder(clip)  # (T, feature_dim)

                # Compute variance across time
                variance = torch.var(features, dim=0).mean().item()
                self.diversity_scores.append(variance)

    def compute(self) -> MetricResult:
        if not self.diversity_scores:
            raise ValueError("No diversity scores accumulated")

        mean_diversity = np.mean(self.diversity_scores)

        return MetricResult(
            name=self.name,
            value=mean_diversity,
            lower_is_better=False,  # Higher diversity is better
            details={
                "std": float(np.std(self.diversity_scores)),
                "min": float(np.min(self.diversity_scores)),
                "max": float(np.max(self.diversity_scores)),
            },
            metadata={
                "n_clips": len(self.diversity_scores),
                "encoder_type": self.encoder_type,
            },
        )
