"""
VersaFace Autoencoder Package

This package provides the autoencoder architecture for VersaFace talking-face generation,
built on top of the Hallo framework.

Key Components:
- VAE: Variational Autoencoder for image-latent conversion
- FaceEncoder: Face embedding extraction and projection
- AudioEncoder: Audio feature extraction and projection
- UNet3D: Main diffusion model with motion modules

Usage:
    from versaface_autoencoder import VersaFaceEncoder, VersaFaceDecoder

    encoder = VersaFaceEncoder(device="cuda")
    decoder = VersaFaceDecoder(device="cuda")

    # Encode reference image
    latent, face_embed = encoder.encode(reference_image)

    # Decode with audio conditioning
    video_frames = decoder.decode(latent, audio_features, face_embed)
"""

from .models.vae_wrapper import VAEWrapper
from .models.face_encoder import FaceEncoderWrapper
from .models.audio_encoder import AudioEncoderWrapper

__all__ = [
    "VAEWrapper",
    "FaceEncoderWrapper",
    "AudioEncoderWrapper",
]

__version__ = "0.1.0"
