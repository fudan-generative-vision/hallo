"""
Face Encoder wrapper for VersaFace.

Combines InsightFace face embedding extraction with the ImageProjModel
for projecting embeddings to cross-attention space.

Architecture:
    Face Image → InsightFace → 512-dim → ImageProj → 4×1024 tokens
"""

from typing import Optional, Tuple, Union, List
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class ImageProjModel(nn.Module):
    """
    Projects face embeddings to cross-attention context tokens.

    From Hallo: Projects 512-dim InsightFace embeddings to (num_tokens × cross_attention_dim).

    Args:
        cross_attention_dim: UNet cross-attention dimension (typically 1024)
        clip_embeddings_dim: Input embedding dimension (512 for InsightFace)
        num_tokens: Number of output context tokens (default 4)
    """

    def __init__(
        self,
        cross_attention_dim: int = 1024,
        clip_embeddings_dim: int = 512,
        num_tokens: int = 4,
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.num_tokens = num_tokens

        # Project to (num_tokens × cross_attention_dim)
        self.proj = nn.Linear(
            clip_embeddings_dim,
            num_tokens * cross_attention_dim,
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Project face embeddings to context tokens.

        Args:
            image_embeds: (batch_size, clip_embeddings_dim) face embeddings

        Returns:
            (batch_size, num_tokens, cross_attention_dim) context tokens
        """
        batch_size = image_embeds.shape[0]

        # Project
        clip_extra_context_tokens = self.proj(image_embeds)

        # Reshape to tokens
        clip_extra_context_tokens = clip_extra_context_tokens.reshape(
            batch_size, self.num_tokens, self.cross_attention_dim
        )

        # Normalize
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)

        return clip_extra_context_tokens


class FaceEncoderWrapper(nn.Module):
    """
    Complete face encoding pipeline for VersaFace.

    Combines:
    1. InsightFace for face embedding extraction (512-dim)
    2. ImageProjModel for cross-attention projection (4×1024)

    Args:
        face_analysis_path: Path to InsightFace models
        cross_attention_dim: UNet cross-attention dimension
        num_tokens: Number of context tokens to generate
        device: Computation device
    """

    def __init__(
        self,
        face_analysis_path: Optional[str] = None,
        cross_attention_dim: int = 1024,
        num_tokens: int = 4,
        device: str = "cuda",
    ):
        super().__init__()

        self.face_analysis_path = face_analysis_path
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.device = device

        # Initialize projection model
        self.image_proj = ImageProjModel(
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=512,  # InsightFace output dim
            num_tokens=num_tokens,
        )
        self.image_proj.to(device)

        # Face analysis (lazy loaded)
        self._face_analysis = None

    def _load_face_analysis(self):
        """Lazily load InsightFace model."""
        if self._face_analysis is not None:
            return

        try:
            from insightface.app import FaceAnalysis

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            self._face_analysis = FaceAnalysis(
                name="",
                root=self.face_analysis_path,
                providers=providers if self.device == "cuda" else ["CPUExecutionProvider"],
            )
            self._face_analysis.prepare(ctx_id=0 if self.device == "cuda" else -1)

        except ImportError:
            raise ImportError(
                "insightface package required. "
                "Install with: pip install insightface onnxruntime-gpu"
            )

    @property
    def face_analysis(self):
        """Get InsightFace model."""
        self._load_face_analysis()
        return self._face_analysis

    @property
    def embedding_dim(self) -> int:
        """Face embedding dimension (512)."""
        return 512

    def extract_face_embedding(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Extract face embedding from a single image using InsightFace.

        Args:
            image: RGB image as numpy array (H, W, 3) or tensor (C, H, W)

        Returns:
            (512,) face embedding vector
        """
        self._load_face_analysis()

        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:  # C, H, W
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()

        # Convert to uint8 if float
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Convert RGB to BGR for InsightFace
        image_bgr = image[:, :, ::-1]

        # Detect faces
        faces = self._face_analysis.get(image_bgr)

        if not faces:
            # Return zero embedding if no face detected
            return np.zeros(512, dtype=np.float32)

        # Get largest face
        largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

        return largest_face.normed_embedding

    def extract_batch_embeddings(
        self,
        images: Union[List[np.ndarray], torch.Tensor],
    ) -> torch.Tensor:
        """
        Extract face embeddings from a batch of images.

        Args:
            images: List of images or tensor (N, C, H, W)

        Returns:
            (N, 512) face embedding tensor
        """
        if isinstance(images, torch.Tensor):
            images = [images[i] for i in range(images.shape[0])]

        embeddings = []
        for img in images:
            emb = self.extract_face_embedding(img)
            embeddings.append(emb)

        return torch.tensor(np.stack(embeddings), dtype=torch.float32, device=self.device)

    def forward(
        self,
        images: Union[List[np.ndarray], torch.Tensor],
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full face encoding pipeline.

        Args:
            images: List of images or tensor (N, C, H, W)
            return_embeddings: If True, also return raw face embeddings

        Returns:
            (N, num_tokens, cross_attention_dim) context tokens
            If return_embeddings: also returns (N, 512) embeddings
        """
        # Extract embeddings
        face_embeddings = self.extract_batch_embeddings(images)

        # Project to context tokens
        context_tokens = self.image_proj(face_embeddings)

        if return_embeddings:
            return context_tokens, face_embeddings
        return context_tokens

    def encode_reference(
        self,
        reference_image: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode a single reference image for generation.

        Args:
            reference_image: Reference face image

        Returns:
            (1, num_tokens, cross_attention_dim) context tokens
        """
        if isinstance(reference_image, np.ndarray):
            reference_image = [reference_image]
        elif isinstance(reference_image, torch.Tensor):
            if reference_image.dim() == 3:
                reference_image = reference_image.unsqueeze(0)

        return self.forward(reference_image)


class FaceLocatorWrapper(nn.Module):
    """
    Wrapper for FaceLocator from Hallo.

    Encodes face masks to conditioning features using 3D convolutions.

    The FaceLocator produces multi-scale conditioning that guides
    the diffusion model to focus on face regions.
    """

    def __init__(
        self,
        conditioning_embedding_channels: int = 320,
        conditioning_channels: int = 3,
        device: str = "cuda",
    ):
        super().__init__()

        self.conditioning_embedding_channels = conditioning_embedding_channels
        self.conditioning_channels = conditioning_channels
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazily load FaceLocator from Hallo."""
        if self._model is not None:
            return

        try:
            from hallo.models.face_locator import FaceLocator

            self._model = FaceLocator(
                conditioning_embedding_channels=self.conditioning_embedding_channels,
                conditioning_channels=self.conditioning_channels,
            )
            self._model.to(self.device)
            self._model.eval()

        except ImportError:
            raise ImportError(
                "Could not import FaceLocator from hallo.models. "
                "Ensure the hallo package is properly installed."
            )

    def forward(
        self,
        face_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode face mask to conditioning features.

        Args:
            face_mask: (N, C, T, H, W) face mask tensor

        Returns:
            (N, conditioning_embedding_channels, T, H', W') conditioning features
        """
        self._load_model()

        face_mask = face_mask.to(device=self.device)

        with torch.no_grad():
            conditioning = self._model(face_mask)

        return conditioning


def test_face_encoder():
    """Test FaceEncoderWrapper functionality."""
    print("Testing FaceEncoderWrapper...")

    # Initialize (without InsightFace for testing)
    encoder = FaceEncoderWrapper(device="cpu")

    # Test ImageProjModel directly
    proj = ImageProjModel(cross_attention_dim=1024, clip_embeddings_dim=512, num_tokens=4)

    fake_embedding = torch.randn(2, 512)
    tokens = proj(fake_embedding)

    print(f"Input embedding shape: {fake_embedding.shape}")
    print(f"Output tokens shape: {tokens.shape}")
    assert tokens.shape == (2, 4, 1024), f"Unexpected shape: {tokens.shape}"

    print("FaceEncoderWrapper tests passed!")


if __name__ == "__main__":
    test_face_encoder()
