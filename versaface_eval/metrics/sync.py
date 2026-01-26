"""
SyncNet-based lip synchronization metrics.

Computes:
- Sync-C (= LSE-C, Lip Sync Error - Confidence): Higher is better
- Sync-D (= LSE-D, Lip Sync Error - Distance): Lower is better

IMPORTANT: These metrics have POOR correlation with human evaluation!
According to THEval (arXiv 2511.04520):
- LSE-C correlation with human preference: ρ = -0.164
- LSE-D correlation with human preference: ρ = -0.269

Consider supplementing with THEval metrics for more reliable evaluation.

Verified Implementation:
The canonical implementation is joonson/syncnet_python:
    https://github.com/joonson/syncnet_python

This module provides two options:
1. SyncNetWrapper: Wraps the official syncnet_python (recommended)
2. SyncNetCalculator: Standalone implementation (fallback)

Sources:
- Wav2Lip Paper: https://arxiv.org/pdf/2008.10010
- SyncNet Python: https://github.com/joonson/syncnet_python
- Confirmed via: https://github.com/Rudrabha/Wav2Lip/issues/284
  - LSE-D = "Min dist" from SyncNet output
  - LSE-C = "Confidence" from SyncNet output
"""

import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MetricCalculator, MetricResult


class SyncNetWrapper(MetricCalculator):
    """
    Wrapper around the official joonson/syncnet_python implementation.

    This is the RECOMMENDED approach for computing Sync-C and Sync-D metrics
    to ensure comparability with published results.

    Requirements:
        1. Clone syncnet_python: git clone https://github.com/joonson/syncnet_python
        2. Set SYNCNET_PATH environment variable or pass syncnet_dir to constructor
        3. Install dependencies: pip install -r requirements.txt (in syncnet_python)

    Usage:
        calculator = SyncNetWrapper(syncnet_dir="/path/to/syncnet_python")
        calculator.add_video("/path/to/video.mp4")
        result = calculator.compute()
        # result.details contains {"sync_c": ..., "sync_d": ...}
    """

    def __init__(
        self,
        syncnet_dir: Optional[str] = None,
        device: str = "cuda",
        tmp_dir: Optional[str] = None,
    ):
        """
        Args:
            syncnet_dir: Path to cloned syncnet_python repository.
                         If None, uses SYNCNET_PATH environment variable.
            device: Computation device (passed to syncnet)
            tmp_dir: Temporary directory for intermediate files
        """
        self.syncnet_dir = syncnet_dir or os.environ.get("SYNCNET_PATH")
        self.tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="syncnet_")

        if self.syncnet_dir is None:
            raise ValueError(
                "syncnet_dir not provided and SYNCNET_PATH not set. "
                "Please clone https://github.com/joonson/syncnet_python "
                "and provide the path."
            )

        if not Path(self.syncnet_dir).exists():
            raise ValueError(f"SyncNet directory not found: {self.syncnet_dir}")

        super().__init__(device)

    @property
    def name(self) -> str:
        return "SyncNet (Official)"

    def reset(self) -> None:
        """Reset accumulated results."""
        self.results: List[Dict[str, float]] = []
        self.video_paths: List[str] = []

    def add_video(self, video_path: str) -> None:
        """
        Add a video to be evaluated.

        Args:
            video_path: Path to video file (MP4, AVI, etc.)
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.video_paths.append(video_path)

    def _run_syncnet_on_video(self, video_path: str) -> Dict[str, float]:
        """
        Run syncnet_python pipeline on a single video.

        Returns dict with:
            - offset: Audio-video offset
            - min_dist: Minimum distance (= LSE-D = Sync-D)
            - confidence: Confidence score (= LSE-C = Sync-C)
        """
        video_name = Path(video_path).stem

        # Run pipeline
        pipeline_cmd = [
            "python", str(Path(self.syncnet_dir) / "run_pipeline.py"),
            "--videofile", video_path,
            "--reference", video_name,
            "--data_dir", self.tmp_dir,
        ]

        try:
            subprocess.run(pipeline_cmd, check=True, capture_output=True, cwd=self.syncnet_dir)
        except subprocess.CalledProcessError as e:
            print(f"SyncNet pipeline failed: {e.stderr.decode()}")
            return {"offset": 0, "min_dist": float('inf'), "confidence": 0}

        # Run syncnet evaluation
        syncnet_cmd = [
            "python", str(Path(self.syncnet_dir) / "run_syncnet.py"),
            "--data_dir", self.tmp_dir,
            "--reference", video_name,
        ]

        try:
            result = subprocess.run(
                syncnet_cmd, check=True, capture_output=True,
                cwd=self.syncnet_dir, text=True
            )
            output = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"SyncNet evaluation failed: {e.stderr}")
            return {"offset": 0, "min_dist": float('inf'), "confidence": 0}

        # Parse output: "AV offset: X, Min dist: Y, Confidence: Z"
        try:
            parts = output.strip().split(", ")
            offset = int(parts[0].split(": ")[1])
            min_dist = float(parts[1].split(": ")[1])
            confidence = float(parts[2].split(": ")[1])
            return {"offset": offset, "min_dist": min_dist, "confidence": confidence}
        except (IndexError, ValueError) as e:
            print(f"Failed to parse SyncNet output: {output}")
            return {"offset": 0, "min_dist": float('inf'), "confidence": 0}

    def update(self, video_path: Optional[str] = None, **kwargs) -> None:
        """
        Process a video and accumulate results.

        Args:
            video_path: Path to video file
        """
        if video_path:
            self.add_video(video_path)

    def compute(self) -> MetricResult:
        """
        Compute Sync-C and Sync-D from all accumulated videos.

        Returns:
            MetricResult with:
                - value: Sync-C (primary metric, higher is better)
                - details["sync_c"]: Mean LSE-C (Confidence)
                - details["sync_d"]: Mean LSE-D (Min dist)
        """
        if not self.video_paths:
            raise ValueError("No videos added for evaluation")

        all_results = []
        for video_path in self.video_paths:
            result = self._run_syncnet_on_video(video_path)
            all_results.append(result)
            self.results.append(result)

        # Compute means
        sync_c_values = [r["confidence"] for r in all_results]
        sync_d_values = [r["min_dist"] for r in all_results]

        mean_sync_c = np.mean(sync_c_values)
        mean_sync_d = np.mean(sync_d_values)

        return MetricResult(
            name="Sync",
            value=mean_sync_c,  # Report Sync-C as primary
            lower_is_better=False,  # Higher Sync-C is better
            details={
                "sync_c": mean_sync_c,
                "sync_d": mean_sync_d,
                "sync_c_std": float(np.std(sync_c_values)),
                "sync_d_std": float(np.std(sync_d_values)),
                "per_video": all_results,
            },
            metadata={
                "n_videos": len(self.video_paths),
                "implementation": "joonson/syncnet_python",
                "warning": "Sync metrics have poor correlation with human evaluation (ρ ≈ -0.2)",
            },
        )


class SyncNetCalculator(MetricCalculator):
    """
    Standalone SyncNet-style lip sync calculator.

    This is a FALLBACK implementation when syncnet_python is not available.
    For publishable results, use SyncNetWrapper with the official implementation.

    Note: This implementation may not produce identical results to syncnet_python.
    """

    def __init__(
        self,
        device: str = "cuda",
        syncnet_checkpoint: Optional[str] = None,
    ):
        """
        Args:
            device: Computation device
            syncnet_checkpoint: Path to SyncNet weights (optional)
        """
        self.syncnet_checkpoint = syncnet_checkpoint
        super().__init__(device)

        # Initialize model
        self.syncnet = SyncNetModel(device)
        if syncnet_checkpoint and Path(syncnet_checkpoint).exists():
            self.syncnet.load_weights(syncnet_checkpoint)
        self.syncnet.eval()

    @property
    def name(self) -> str:
        return "SyncNet (Standalone)"

    def reset(self) -> None:
        """Reset accumulated scores."""
        self.sync_scores: List[Dict[str, float]] = []

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

        # Process in windows
        N, T = gen_frames.shape[:2]
        window_size = 5
        fps = 25.0

        with torch.no_grad():
            for i in range(0, T - window_size + 1, window_size):
                face_window = lip_crops[:, i:i + window_size]

                audio_start = int(i / fps * 100)
                audio_end = int((i + window_size) / fps * 100)
                mel_window = mel[:, :, :, audio_start:audio_end]

                if mel_window.size(-1) < 20:
                    continue

                face_embed, audio_embed = self.syncnet(
                    face_window.to(self.device),
                    mel_window.to(self.device),
                )

                scores = self._compute_sync_score(face_embed, audio_embed)
                self.sync_scores.append(scores)

    def _extract_lip_region(
        self,
        frames: torch.Tensor,
        lip_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract lip region crops from video frames."""
        N, T, C, H, W = frames.shape

        if lip_coords is not None:
            crops = []
            for i in range(N):
                x1, y1, x2, y2 = lip_coords[i].int().tolist()
                crop = frames[i, :, :, y1:y2, x1:x2]
                crop = F.interpolate(crop, size=(96, 192), mode='bilinear', align_corners=False)
                crops.append(crop)
            return torch.stack(crops)
        else:
            # Default: lower half of face
            lip_start_y = int(H * 0.5)
            lip_end_y = int(H * 0.9)
            lip_start_x = int(W * 0.25)
            lip_end_x = int(W * 0.75)

            crops = frames[:, :, :, lip_start_y:lip_end_y, lip_start_x:lip_end_x]
            crops = crops.view(N * T, C, crops.size(-2), crops.size(-1))
            crops = F.interpolate(crops, size=(96, 192), mode='bilinear', align_corners=False)
            crops = crops.view(N, T, C, 96, 192)

            return crops

    def _compute_mel_spectrogram(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Compute mel spectrogram from audio."""
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

            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            if mel.dim() == 3:
                mel = mel.unsqueeze(1)

            return mel

        except ImportError:
            raise ImportError("torchaudio required for mel spectrogram computation")

    def _compute_sync_score(
        self,
        face_embed: torch.Tensor,
        audio_embed: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute sync scores from embeddings."""
        # Cosine similarity -> Sync-C (higher is better)
        cos_sim = F.cosine_similarity(face_embed, audio_embed, dim=1)
        sync_c = cos_sim.mean().item()

        # L2 distance -> Sync-D (lower is better)
        l2_dist = torch.norm(face_embed - audio_embed, p=2, dim=1)
        sync_d = l2_dist.mean().item()

        return {"sync_c": sync_c, "sync_d": sync_d}

    def compute(self) -> MetricResult:
        """Compute final sync metrics."""
        if not self.sync_scores:
            raise ValueError("No sync scores accumulated")

        sync_c_values = [s["sync_c"] for s in self.sync_scores]
        sync_d_values = [s["sync_d"] for s in self.sync_scores]

        mean_sync_c = np.mean(sync_c_values)
        mean_sync_d = np.mean(sync_d_values)

        return MetricResult(
            name="Sync",
            value=mean_sync_c,
            lower_is_better=False,
            details={
                "sync_c": mean_sync_c,
                "sync_d": mean_sync_d,
                "sync_c_std": float(np.std(sync_c_values)),
                "sync_d_std": float(np.std(sync_d_values)),
            },
            metadata={
                "n_windows": len(self.sync_scores),
                "implementation": "standalone (may differ from official)",
                "warning": "For publishable results, use SyncNetWrapper with joonson/syncnet_python",
            },
        )


class SyncNetModel(nn.Module):
    """
    SyncNet model architecture.

    Note: This is a simplified version. For exact replication,
    use the official syncnet_python implementation.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

        # Face encoder
        self.face_encoder = nn.Sequential(
            nn.Conv2d(15, 96, kernel_size=7, stride=1, padding=3),
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
            nn.Linear(256 * 3 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )

        # Audio encoder
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
            nn.Linear(256 * 3 * 5, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )

        self.to(device)

    def load_weights(self, checkpoint_path: str) -> None:
        """Load pretrained weights."""
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        face_crops: torch.Tensor,
        mel_spec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings."""
        N = face_crops.size(0)
        face_crops = face_crops.view(N, -1, face_crops.size(-2), face_crops.size(-1))

        face_embed = self.face_encoder(face_crops)
        audio_embed = self.audio_encoder(mel_spec)

        face_embed = F.normalize(face_embed, p=2, dim=1)
        audio_embed = F.normalize(audio_embed, p=2, dim=1)

        return face_embed, audio_embed


# Convenience function
def compute_sync_metrics(
    video_path: str,
    syncnet_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute Sync-C and Sync-D for a single video.

    Args:
        video_path: Path to video file
        syncnet_dir: Path to syncnet_python (optional, uses SYNCNET_PATH env var if not provided)

    Returns:
        Dict with "sync_c" and "sync_d" values
    """
    calculator = SyncNetWrapper(syncnet_dir=syncnet_dir)
    calculator.add_video(video_path)
    result = calculator.compute()
    return {
        "sync_c": result.details["sync_c"],
        "sync_d": result.details["sync_d"],
    }
