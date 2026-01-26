"""
SyncNet-based lip synchronization metrics.

Computes:
- Sync-C (Sync Confidence): Average confidence of audio-visual sync
- Sync-D (Sync Distance): Average distance/offset between audio and visual

Implementation follows wav2lip/syncnet conventions for comparability.
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MetricCalculator, MetricResult


class SyncNetModel(nn.Module):
    """
    SyncNet model for audio-visual sync scoring.

    Architecture from "Out of time: automated lip sync in the wild" (Chung & Zisserman).

    Outputs embeddings for audio and video that should be close when in sync.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

        # Face encoder (processes lip region crops)
        self.face_encoder = nn.Sequential(
            nn.Conv2d(15, 96, kernel_size=7, stride=1, padding=3),  # 5 frames * 3 channels
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Flatten(),
            nn.Linear(256 * 3 * 6, 512),  # Adjust based on input size
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )

        # Audio encoder (processes mel spectrograms)
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1),

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Flatten(),
            nn.Linear(256 * 3 * 5, 512),  # Adjust based on mel size
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )

        self.to(device)
        self._weights_loaded = False

    def load_weights(self, checkpoint_path: str) -> None:
        """Load pretrained SyncNet weights."""
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)
        self._weights_loaded = True

    def forward(
        self,
        face_crops: torch.Tensor,
        mel_spec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract audio and visual embeddings.

        Args:
            face_crops: (N, 5, 3, H, W) lip region crops (5 consecutive frames)
            mel_spec: (N, 1, mel_bins, mel_frames) mel spectrogram

        Returns:
            (face_embed, audio_embed) both of shape (N, 512)
        """
        # Reshape face crops to (N, 15, H, W) - concatenate channels
        N = face_crops.size(0)
        face_crops = face_crops.view(N, -1, face_crops.size(-2), face_crops.size(-1))

        face_embed = self.face_encoder(face_crops)
        audio_embed = self.audio_encoder(mel_spec)

        # L2 normalize
        face_embed = F.normalize(face_embed, p=2, dim=1)
        audio_embed = F.normalize(audio_embed, p=2, dim=1)

        return face_embed, audio_embed


class SyncNetCalculator(MetricCalculator):
    """
    SyncNet-based lip sync metrics calculator.

    Computes:
    - Sync-C (Confidence): Mean cosine similarity between audio/video embeddings
      Higher = better sync
    - Sync-D (Distance): Mean L2 distance between embeddings
      Lower = better sync

    Args:
        device: Computation device
        syncnet_checkpoint: Path to pretrained SyncNet weights (optional)
        mel_step_size: Step size for mel spectrogram window
        fps: Video FPS for audio alignment
    """

    def __init__(
        self,
        device: str = "cuda",
        syncnet_checkpoint: Optional[str] = None,
        mel_step_size: int = 16,
        fps: float = 25.0,
    ):
        self.mel_step_size = mel_step_size
        self.fps = fps
        self.syncnet_checkpoint = syncnet_checkpoint

        super().__init__(device)

        # Initialize SyncNet
        self.syncnet = SyncNetModel(device)
        if syncnet_checkpoint:
            self.syncnet.load_weights(syncnet_checkpoint)
        self.syncnet.eval()

    @property
    def name(self) -> str:
        return "SyncNet"

    def reset(self) -> None:
        """Reset accumulated scores."""
        self.sync_scores: List[Dict[str, float]] = []

    def _extract_lip_region(
        self,
        frames: torch.Tensor,
        lip_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract lip region crops from video frames.

        Args:
            frames: (N, T, C, H, W) video frames
            lip_coords: Optional (N, 4) lip bounding box coordinates

        Returns:
            (N, T, C, lip_H, lip_W) lip region crops
        """
        N, T, C, H, W = frames.shape

        if lip_coords is not None:
            # Use provided lip coordinates
            crops = []
            for i in range(N):
                x1, y1, x2, y2 = lip_coords[i].int().tolist()
                crop = frames[i, :, :, y1:y2, x1:x2]
                crop = F.interpolate(crop, size=(96, 192), mode='bilinear', align_corners=False)
                crops.append(crop)
            return torch.stack(crops)
        else:
            # Default: assume lower third of face contains lips
            lip_start_y = int(H * 0.5)
            lip_end_y = int(H * 0.9)
            lip_start_x = int(W * 0.25)
            lip_end_x = int(W * 0.75)

            crops = frames[:, :, :, lip_start_y:lip_end_y, lip_start_x:lip_end_x]
            # Resize to expected dimensions
            crops = crops.view(N * T, C, crops.size(-2), crops.size(-1))
            crops = F.interpolate(crops, size=(96, 192), mode='bilinear', align_corners=False)
            crops = crops.view(N, T, C, 96, 192)

            return crops

    def _compute_mel_spectrogram(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Compute mel spectrogram from audio.

        Args:
            audio: (N, audio_samples) or (N, 1, audio_samples) audio tensor
            sample_rate: Audio sample rate

        Returns:
            (N, 1, mel_bins, mel_frames) mel spectrogram
        """
        try:
            import torchaudio
            import torchaudio.transforms as T

            if audio.dim() == 3:
                audio = audio.squeeze(1)

            mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=160,
                n_mels=80,
            ).to(audio.device)

            mel = mel_transform(audio)
            mel = torch.log(mel.clamp(min=1e-5))

            # Add channel dimension
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            if mel.dim() == 3:
                mel = mel.unsqueeze(1)

            return mel

        except ImportError:
            # Fallback using numpy/librosa
            import librosa

            mels = []
            for i in range(audio.size(0)):
                audio_np = audio[i].cpu().numpy()
                if audio_np.ndim > 1:
                    audio_np = audio_np.flatten()

                mel = librosa.feature.melspectrogram(
                    y=audio_np,
                    sr=sample_rate,
                    n_fft=1024,
                    hop_length=160,
                    n_mels=80,
                )
                mel = np.log(np.maximum(mel, 1e-5))
                mels.append(mel)

            mels = np.stack(mels)
            return torch.from_numpy(mels).unsqueeze(1).float()

    def _compute_sync_score(
        self,
        face_embed: torch.Tensor,
        audio_embed: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute sync scores from embeddings.

        Args:
            face_embed: (N, 512) face embeddings
            audio_embed: (N, 512) audio embeddings

        Returns:
            Dict with sync_c (confidence) and sync_d (distance)
        """
        # Cosine similarity (Sync-C)
        cos_sim = F.cosine_similarity(face_embed, audio_embed, dim=1)
        sync_c = cos_sim.mean().item()

        # L2 distance (Sync-D)
        l2_dist = torch.norm(face_embed - audio_embed, p=2, dim=1)
        sync_d = l2_dist.mean().item()

        return {"sync_c": sync_c, "sync_d": sync_d}

    def update(
        self,
        gen_frames: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        lip_coords: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """
        Update with generated video and audio.

        Args:
            gen_frames: (N, T, C, H, W) generated video frames
            audio: (N, audio_samples) audio tensor
            lip_coords: Optional (N, 4) lip bounding box coordinates
        """
        if gen_frames is None or audio is None:
            return

        gen_frames = gen_frames.to(self.device)
        audio = audio.to(self.device)

        # Extract lip regions
        lip_crops = self._extract_lip_region(gen_frames, lip_coords)

        # Compute mel spectrogram
        mel = self._compute_mel_spectrogram(audio)

        # Process in windows of 5 frames
        N, T = gen_frames.shape[:2]
        window_size = 5

        with torch.no_grad():
            for i in range(0, T - window_size + 1, window_size):
                # Get 5-frame window
                face_window = lip_crops[:, i:i + window_size]

                # Get corresponding audio window (approximately)
                audio_start = int(i / self.fps * 100)  # 100 mel frames per second at 16kHz
                audio_end = int((i + window_size) / self.fps * 100)
                mel_window = mel[:, :, :, audio_start:audio_end]

                # Ensure correct dimensions
                if mel_window.size(-1) < 20:  # Minimum mel frames
                    continue

                # Extract embeddings
                face_embed, audio_embed = self.syncnet(
                    face_window.to(self.device),
                    mel_window.to(self.device),
                )

                # Compute scores
                scores = self._compute_sync_score(face_embed, audio_embed)
                self.sync_scores.append(scores)

    def compute(self) -> MetricResult:
        """
        Compute final sync metrics.

        Returns:
            MetricResult with Sync-C and Sync-D values
        """
        if not self.sync_scores:
            raise ValueError("No sync scores accumulated")

        sync_c_values = [s["sync_c"] for s in self.sync_scores]
        sync_d_values = [s["sync_d"] for s in self.sync_scores]

        mean_sync_c = np.mean(sync_c_values)
        mean_sync_d = np.mean(sync_d_values)

        # Primary metric is Sync-C (higher is better)
        return MetricResult(
            name="Sync",
            value=mean_sync_c,  # Report Sync-C as primary
            lower_is_better=False,  # Higher Sync-C is better
            details={
                "sync_c": mean_sync_c,
                "sync_d": mean_sync_d,
                "sync_c_std": float(np.std(sync_c_values)),
                "sync_d_std": float(np.std(sync_d_values)),
            },
            metadata={
                "n_windows": len(self.sync_scores),
                "syncnet_checkpoint": self.syncnet_checkpoint,
            },
        )

    def get_sync_c(self) -> float:
        """Get Sync-C score."""
        result = self.compute()
        return result.details["sync_c"]

    def get_sync_d(self) -> float:
        """Get Sync-D score."""
        result = self.compute()
        return result.details["sync_d"]
