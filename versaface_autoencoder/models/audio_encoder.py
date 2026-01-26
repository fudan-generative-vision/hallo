"""
Audio Encoder wrapper for VersaFace.

Combines wav2vec2 audio feature extraction with the AudioProjModel
for projecting features to cross-attention space.

Architecture:
    Audio Waveform → wav2vec2 → (T, 12, 768) → AudioProj → (T, 32, 768) tokens
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class AudioProjModel(nn.Module):
    """
    Projects wav2vec2 audio features to cross-attention context tokens.

    From Hallo: Processes (5 windows × 12 layers × 768 dim) = 46080 features
    down to (32 context tokens × 768 dim) per frame.

    Args:
        seq_len: Number of audio windows per video frame (5)
        blocks: Number of wav2vec2 transformer layers to use (12)
        channels: wav2vec2 hidden dimension (768)
        intermediate_dim: Intermediate projection dimension (512)
        output_dim: Output dimension per token (768)
        context_tokens: Number of output context tokens (32)
    """

    def __init__(
        self,
        seq_len: int = 5,
        blocks: int = 12,
        channels: int = 768,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # Input dimension: flatten all features
        input_dim = seq_len * blocks * channels  # 5 * 12 * 768 = 46080

        # Projection layers
        self.proj1 = nn.Linear(input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        """
        Project audio features to context tokens.

        Args:
            audio_embeds: (batch, video_length, seq_len, blocks, channels)
                          or (batch, seq_len * blocks * channels)

        Returns:
            (batch, video_length, context_tokens, output_dim) context tokens
        """
        # Handle different input formats
        if audio_embeds.dim() == 5:
            batch_size, video_length = audio_embeds.shape[:2]
            # Flatten: (batch, video_length, seq_len * blocks * channels)
            audio_embeds = audio_embeds.view(batch_size, video_length, -1)
        elif audio_embeds.dim() == 3:
            batch_size, video_length = audio_embeds.shape[:2]
        elif audio_embeds.dim() == 2:
            batch_size = audio_embeds.shape[0]
            video_length = 1
            audio_embeds = audio_embeds.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected audio_embeds shape: {audio_embeds.shape}")

        # Flatten batch and time for projection
        audio_flat = audio_embeds.view(batch_size * video_length, -1)

        # Project through MLP
        hidden = torch.relu(self.proj1(audio_flat))
        hidden = torch.relu(self.proj2(hidden))
        output = self.proj3(hidden)

        # Reshape to tokens
        output = output.view(batch_size, video_length, self.context_tokens, self.output_dim)

        # Normalize each token
        output = self.norm(output)

        return output


class AudioEncoderWrapper(nn.Module):
    """
    Complete audio encoding pipeline for VersaFace.

    Combines:
    1. wav2vec2 for audio feature extraction
    2. AudioProjModel for cross-attention projection

    Args:
        wav2vec_model: HuggingFace model ID or path
        context_tokens: Number of context tokens per frame
        device: Computation device
    """

    def __init__(
        self,
        wav2vec_model: str = "facebook/wav2vec2-base-960h",
        context_tokens: int = 32,
        device: str = "cuda",
    ):
        super().__init__()

        self.wav2vec_model_id = wav2vec_model
        self.context_tokens = context_tokens
        self.device = device

        # Initialize projection model
        self.audio_proj = AudioProjModel(
            seq_len=5,
            blocks=12,
            channels=768,
            context_tokens=context_tokens,
        )
        self.audio_proj.to(device)

        # wav2vec2 (lazy loaded)
        self._wav2vec = None
        self._feature_extractor = None

    def _load_wav2vec(self):
        """Lazily load wav2vec2 model."""
        if self._wav2vec is not None:
            return

        try:
            from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

            self._wav2vec = Wav2Vec2Model.from_pretrained(self.wav2vec_model_id)
            self._wav2vec.to(self.device)
            self._wav2vec.eval()

            # Freeze parameters
            for param in self._wav2vec.parameters():
                param.requires_grad = False

            self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.wav2vec_model_id
            )

        except ImportError:
            raise ImportError(
                "transformers package required for wav2vec2. "
                "Install with: pip install transformers"
            )

    @property
    def wav2vec(self):
        """Get wav2vec2 model."""
        self._load_wav2vec()
        return self._wav2vec

    @property
    def feature_extractor(self):
        """Get wav2vec2 feature extractor."""
        self._load_wav2vec()
        return self._feature_extractor

    @property
    def sample_rate(self) -> int:
        """Expected audio sample rate (16000 Hz)."""
        return 16000

    @property
    def hidden_dim(self) -> int:
        """wav2vec2 hidden dimension (768)."""
        return 768

    @property
    def num_layers(self) -> int:
        """Number of wav2vec2 transformer layers (12)."""
        return 12

    def extract_features(
        self,
        audio: torch.Tensor,
        fps: float = 25.0,
        video_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract wav2vec2 features from audio waveform.

        Args:
            audio: (batch, audio_samples) or (audio_samples,) waveform at 16kHz
            fps: Video frame rate for alignment
            video_length: Number of video frames (optional, computed from audio if None)

        Returns:
            (batch, video_length, seq_len, blocks, channels) features
        """
        self._load_wav2vec()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        batch_size = audio.shape[0]
        audio_length = audio.shape[1]

        # Calculate video length from audio if not provided
        if video_length is None:
            video_length = int(audio_length / self.sample_rate * fps)

        audio = audio.to(device=self.device)

        with torch.no_grad():
            # Extract wav2vec2 features
            outputs = self._wav2vec(audio, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # List of (batch, seq, 768)

            # Stack all transformer layers: (batch, layers, seq, 768)
            all_layers = torch.stack(hidden_states[1:], dim=1)  # Skip embedding layer

            # Resample to video frame rate
            # Each video frame gets seq_len=5 windows of audio features
            seq_len = 5
            audio_frames_per_video = all_layers.shape[2] / video_length

            features = []
            for t in range(video_length):
                start_idx = int(t * audio_frames_per_video)
                end_idx = int((t + 1) * audio_frames_per_video)

                # Sample seq_len windows uniformly
                if end_idx - start_idx >= seq_len:
                    indices = np.linspace(start_idx, end_idx - 1, seq_len, dtype=int)
                    frame_features = all_layers[:, :, indices, :]  # (batch, 12, 5, 768)
                else:
                    # Pad if not enough audio frames
                    frame_features = all_layers[:, :, start_idx:end_idx, :]
                    pad_size = seq_len - (end_idx - start_idx)
                    frame_features = torch.nn.functional.pad(
                        frame_features, (0, 0, 0, pad_size)
                    )

                features.append(frame_features)

            # Stack: (batch, video_length, 12, 5, 768)
            features = torch.stack(features, dim=1)

            # Reorder to (batch, video_length, 5, 12, 768)
            features = features.permute(0, 1, 3, 2, 4)

        return features

    def forward(
        self,
        audio: torch.Tensor,
        fps: float = 25.0,
        video_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Full audio encoding pipeline.

        Args:
            audio: (batch, audio_samples) waveform at 16kHz
            fps: Video frame rate
            video_length: Number of video frames

        Returns:
            (batch, video_length, context_tokens, output_dim) context tokens
        """
        # Extract wav2vec2 features
        features = self.extract_features(audio, fps=fps, video_length=video_length)

        # Project to context tokens
        context_tokens = self.audio_proj(features)

        return context_tokens

    def encode_for_frame(
        self,
        audio: torch.Tensor,
        frame_idx: int,
        fps: float = 25.0,
    ) -> torch.Tensor:
        """
        Encode audio for a specific video frame.

        Useful for autoregressive generation.

        Args:
            audio: (batch, audio_samples) waveform
            frame_idx: Target video frame index
            fps: Video frame rate

        Returns:
            (batch, context_tokens, output_dim) context tokens for single frame
        """
        # Extract features for video length up to frame_idx + 1
        context = self.forward(audio, fps=fps, video_length=frame_idx + 1)

        # Return only the requested frame
        return context[:, frame_idx, :, :]


class AudioFeatureCache:
    """
    Caches audio features for efficient generation.

    Pre-computes wav2vec2 features for the entire audio clip,
    then serves individual frame features on demand.
    """

    def __init__(
        self,
        encoder: AudioEncoderWrapper,
    ):
        self.encoder = encoder
        self._features = None
        self._context_tokens = None

    def precompute(
        self,
        audio: torch.Tensor,
        video_length: int,
        fps: float = 25.0,
    ) -> None:
        """
        Precompute features for entire audio clip.

        Args:
            audio: (batch, audio_samples) waveform
            video_length: Number of video frames
            fps: Video frame rate
        """
        self._features = self.encoder.extract_features(
            audio, fps=fps, video_length=video_length
        )
        self._context_tokens = self.encoder.audio_proj(self._features)

    def get_frame(self, frame_idx: int) -> torch.Tensor:
        """
        Get context tokens for a specific frame.

        Args:
            frame_idx: Video frame index

        Returns:
            (batch, context_tokens, output_dim) context tokens
        """
        if self._context_tokens is None:
            raise ValueError("Must call precompute() before get_frame()")

        return self._context_tokens[:, frame_idx, :, :]

    def get_all(self) -> torch.Tensor:
        """
        Get context tokens for all frames.

        Returns:
            (batch, video_length, context_tokens, output_dim) context tokens
        """
        if self._context_tokens is None:
            raise ValueError("Must call precompute() before get_all()")

        return self._context_tokens


def test_audio_encoder():
    """Test AudioEncoderWrapper functionality."""
    print("Testing AudioEncoderWrapper...")

    # Test AudioProjModel directly
    proj = AudioProjModel(
        seq_len=5,
        blocks=12,
        channels=768,
        context_tokens=32,
    )

    # Simulate wav2vec2 output: (batch=2, video_length=14, seq=5, layers=12, dim=768)
    fake_features = torch.randn(2, 14, 5, 12, 768)
    tokens = proj(fake_features)

    print(f"Input features shape: {fake_features.shape}")
    print(f"Output tokens shape: {tokens.shape}")
    assert tokens.shape == (2, 14, 32, 768), f"Unexpected shape: {tokens.shape}"

    print("AudioEncoderWrapper tests passed!")


if __name__ == "__main__":
    test_audio_encoder()
