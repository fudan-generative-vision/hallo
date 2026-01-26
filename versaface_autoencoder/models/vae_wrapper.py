"""
VAE (Variational Autoencoder) wrapper for VersaFace.

Wraps the Diffusers AutoencoderKL (sd-vae-ft-mse) with VersaFace-specific
preprocessing and interface.

The VAE compresses images to a latent space:
- Input: 512×512×3 RGB image
- Output: 64×64×4 latent tensor
- Compression ratio: 8× spatial, 0.75× channels

Architecture based on Stable Diffusion VAE with KL divergence training.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from pathlib import Path


class VAEWrapper(nn.Module):
    """
    Wrapper around Diffusers AutoencoderKL for VersaFace.

    Provides:
    - Lazy loading of pretrained weights
    - Consistent scaling (0.18215 factor)
    - Video-aware encoding/decoding

    Args:
        pretrained_path: Path or HuggingFace model ID
        device: Computation device
        dtype: Model dtype (float16/float32)
    """

    # Standard VAE scaling factor (from Stable Diffusion)
    SCALE_FACTOR = 0.18215

    def __init__(
        self,
        pretrained_path: str = "stabilityai/sd-vae-ft-mse",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.device = device
        self.dtype = dtype
        self._vae = None  # Lazy loading

    def _load_vae(self):
        """Lazily load VAE model."""
        if self._vae is not None:
            return

        try:
            from diffusers import AutoencoderKL

            self._vae = AutoencoderKL.from_pretrained(
                self.pretrained_path,
                torch_dtype=self.dtype,
            )
            self._vae.to(self.device)
            self._vae.eval()

            # Freeze VAE parameters
            for param in self._vae.parameters():
                param.requires_grad = False

        except ImportError:
            raise ImportError(
                "diffusers package required for VAE. "
                "Install with: pip install diffusers"
            )

    @property
    def vae(self):
        """Get the underlying VAE model."""
        self._load_vae()
        return self._vae

    @property
    def latent_channels(self) -> int:
        """Number of latent channels (4 for SD VAE)."""
        return 4

    @property
    def spatial_scale(self) -> int:
        """Spatial downsampling factor (8 for SD VAE)."""
        return 8

    def get_latent_size(
        self,
        image_size: Union[int, Tuple[int, int]],
    ) -> Tuple[int, int]:
        """
        Calculate latent tensor size for given image size.

        Args:
            image_size: Image height/width or (height, width)

        Returns:
            (latent_height, latent_width)
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        return (
            image_size[0] // self.spatial_scale,
            image_size[1] // self.spatial_scale,
        )

    def encode(
        self,
        images: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """
        Encode images to latent space.

        Args:
            images: (N, C, H, W) tensor, values in [0, 1] or [-1, 1]
            sample: If True, sample from distribution; else use mean

        Returns:
            (N, 4, H/8, W/8) latent tensor (scaled)
        """
        self._load_vae()

        # Ensure input is in [-1, 1]
        if images.min() >= 0:
            images = images * 2 - 1

        images = images.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            dist = self._vae.encode(images).latent_dist

            if sample:
                latents = dist.sample()
            else:
                latents = dist.mean

            # Apply scaling factor
            latents = latents * self.SCALE_FACTOR

        return latents

    def encode_video(
        self,
        video: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """
        Encode video frames to latent space.

        Args:
            video: (N, T, C, H, W) tensor, values in [0, 1] or [-1, 1]
            sample: If True, sample from distribution; else use mean

        Returns:
            (N, T, 4, H/8, W/8) latent tensor (scaled)
        """
        N, T, C, H, W = video.shape

        # Reshape to batch of images
        video_flat = video.view(N * T, C, H, W)

        # Encode
        latents_flat = self.encode(video_flat, sample=sample)

        # Reshape back
        _, latent_C, latent_H, latent_W = latents_flat.shape
        latents = latents_flat.view(N, T, latent_C, latent_H, latent_W)

        return latents

    def decode(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latents to images.

        Args:
            latents: (N, 4, H/8, W/8) scaled latent tensor

        Returns:
            (N, C, H, W) tensor, values in [0, 1]
        """
        self._load_vae()

        latents = latents.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            # Undo scaling
            latents = latents / self.SCALE_FACTOR

            # Decode
            images = self._vae.decode(latents).sample

            # Convert from [-1, 1] to [0, 1]
            images = (images / 2 + 0.5).clamp(0, 1)

        return images

    def decode_video(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode video latents to frames.

        Args:
            latents: (N, T, 4, H/8, W/8) scaled latent tensor

        Returns:
            (N, T, C, H, W) tensor, values in [0, 1]
        """
        N, T, latent_C, latent_H, latent_W = latents.shape

        # Reshape to batch of latents
        latents_flat = latents.view(N * T, latent_C, latent_H, latent_W)

        # Decode
        images_flat = self.decode(latents_flat)

        # Reshape back
        _, C, H, W = images_flat.shape
        video = images_flat.view(N, T, C, H, W)

        return video

    def forward(
        self,
        images: torch.Tensor,
        mode: str = "encode",
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Input tensor
            mode: "encode" or "decode"

        Returns:
            Encoded latents or decoded images
        """
        if mode == "encode":
            return self.encode(images)
        elif mode == "decode":
            return self.decode(images)
        else:
            raise ValueError(f"Unknown mode: {mode}")


def test_vae_wrapper():
    """Test VAE wrapper functionality."""
    print("Testing VAEWrapper...")

    # Initialize
    vae = VAEWrapper(device="cpu", dtype=torch.float32)

    # Test with random image
    image = torch.rand(1, 3, 512, 512)

    # Encode
    latent = vae.encode(image, sample=False)
    print(f"Image shape: {image.shape}")
    print(f"Latent shape: {latent.shape}")
    assert latent.shape == (1, 4, 64, 64), f"Unexpected latent shape: {latent.shape}"

    # Decode
    reconstructed = vae.decode(latent)
    print(f"Reconstructed shape: {reconstructed.shape}")
    assert reconstructed.shape == image.shape, f"Shape mismatch: {reconstructed.shape}"

    # Test video
    video = torch.rand(2, 14, 3, 512, 512)
    video_latent = vae.encode_video(video, sample=False)
    print(f"Video shape: {video.shape}")
    print(f"Video latent shape: {video_latent.shape}")
    assert video_latent.shape == (2, 14, 4, 64, 64)

    print("VAEWrapper tests passed!")


if __name__ == "__main__":
    test_vae_wrapper()
