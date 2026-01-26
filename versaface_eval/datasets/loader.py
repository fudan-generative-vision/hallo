"""
Dataset loader for VersaFace evaluation.

Provides PyTorch Dataset interface for loading evaluation samples
with consistent preprocessing matching Hallo conventions.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import json

import torch
from torch.utils.data import Dataset
import numpy as np

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .manifest import EvalSample, EvalManifest


class EvalDataset(Dataset):
    """
    PyTorch Dataset for evaluation samples.

    Loads:
    - Reference frames
    - Audio
    - Ground truth video frames

    Preprocessing matches Hallo conventions:
    - 512x512 resolution (configurable)
    - 25 fps
    - Face-centered crops
    """

    def __init__(
        self,
        manifest: EvalManifest,
        resolution: Tuple[int, int] = (512, 512),
        n_frames: int = 14,
        fps: float = 25.0,
        load_gt_frames: bool = True,
        load_audio: bool = True,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            manifest: EvalManifest containing samples
            resolution: Target resolution (width, height)
            n_frames: Number of frames to load per clip
            fps: Target FPS
            load_gt_frames: Whether to load ground truth video frames
            load_audio: Whether to load audio
            transform: Optional transform to apply to frames
        """
        self.manifest = manifest
        self.resolution = resolution
        self.n_frames = n_frames
        self.fps = fps
        self.load_gt_frames = load_gt_frames
        self.load_audio = load_audio
        self.transform = transform

        # Validate manifest
        self.samples = list(manifest.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        result = {
            "sample_id": sample.sample_id,
            "identity_id": sample.identity_id,
            "dataset": sample.dataset,
        }

        # Load reference frame
        if sample.ref_frame_path and os.path.exists(sample.ref_frame_path):
            ref_frame = self._load_image(sample.ref_frame_path)
            result["ref_frame"] = ref_frame

        # Load ground truth frames
        if self.load_gt_frames and sample.gt_video_path:
            gt_frames = self._load_video_frames(
                sample.gt_video_path,
                start_time=sample.start_time,
                end_time=sample.end_time,
                n_frames=self.n_frames,
            )
            result["gt_frames"] = gt_frames

        # Load audio
        if self.load_audio and sample.audio_path and os.path.exists(sample.audio_path):
            audio = self._load_audio(sample.audio_path)
            result["audio"] = audio

        # Load face mask if available
        if sample.face_mask_path and os.path.exists(sample.face_mask_path):
            mask = self._load_image(sample.face_mask_path, grayscale=True)
            result["face_mask"] = mask

        # Load precomputed embeddings if available
        if sample.face_emb_path and os.path.exists(sample.face_emb_path):
            face_emb = torch.load(sample.face_emb_path, map_location="cpu")
            result["face_emb"] = face_emb

        if sample.audio_emb_path and os.path.exists(sample.audio_emb_path):
            audio_emb = torch.load(sample.audio_emb_path, map_location="cpu")
            result["audio_emb"] = audio_emb

        # Metadata
        result["metadata"] = {
            "fps": sample.fps,
            "start_time": sample.start_time,
            "end_time": sample.end_time,
            "duration": sample.duration,
            "is_silent": sample.is_silent,
            "is_non_english": sample.is_non_english,
            "language": sample.language,
        }

        return result

    def _load_image(
        self,
        path: str,
        grayscale: bool = False,
    ) -> torch.Tensor:
        """Load and preprocess a single image."""
        if not HAS_PIL:
            raise ImportError("PIL is required for image loading")

        img = Image.open(path)
        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        # Resize to target resolution
        img = img.resize(self.resolution, Image.LANCZOS)

        # Convert to tensor
        img_np = np.array(img)
        if grayscale:
            img_tensor = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0
        else:
            img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor

    def _load_video_frames(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float = 0.0,
        n_frames: int = 14,
    ) -> torch.Tensor:
        """
        Load video frames at specified timestamps.

        Returns:
            Tensor of shape (n_frames, C, H, W)
        """
        if not HAS_AV:
            raise ImportError("PyAV is required for video loading")

        container = av.open(video_path)
        stream = container.streams.video[0]

        # Get video properties
        video_fps = float(stream.average_rate)
        duration = float(stream.duration * stream.time_base) if stream.duration else 0

        # Calculate frame indices
        if end_time <= start_time:
            end_time = duration

        clip_duration = end_time - start_time
        frame_times = np.linspace(start_time, end_time, n_frames, endpoint=False)

        frames = []
        for target_time in frame_times:
            # Seek to target time
            target_pts = int(target_time / stream.time_base)
            container.seek(target_pts, stream=stream)

            # Get frame
            for frame in container.decode(video=0):
                if float(frame.pts * stream.time_base) >= target_time:
                    img = frame.to_image().convert("RGB")
                    img = img.resize(self.resolution, Image.LANCZOS)
                    img_np = np.array(img)
                    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
                    frames.append(img_tensor)
                    break

        container.close()

        # Pad if we didn't get enough frames
        while len(frames) < n_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, *self.resolution))

        frames = torch.stack(frames[:n_frames], dim=0)

        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])

        return frames

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file."""
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            return waveform
        except ImportError:
            # Fallback to scipy if torchaudio not available
            try:
                from scipy.io import wavfile
                sample_rate, audio = wavfile.read(audio_path)
                return torch.from_numpy(audio.astype(np.float32))
            except ImportError:
                raise ImportError("Either torchaudio or scipy is required for audio loading")

    def get_sample(self, idx: int) -> EvalSample:
        """Get raw EvalSample without loading data."""
        return self.samples[idx]


class GeneratedDataset(Dataset):
    """
    Dataset for loading generated outputs alongside ground truth.

    Expects output structure:
    {gen_dir}/{sample_id}/
        gen.mp4
        gen_frames/
        audio.wav
        meta.json
    """

    def __init__(
        self,
        manifest: EvalManifest,
        gen_dir: str,
        resolution: Tuple[int, int] = (512, 512),
        n_frames: int = 14,
    ):
        self.manifest = manifest
        self.gen_dir = Path(gen_dir)
        self.resolution = resolution
        self.n_frames = n_frames

        self.samples = list(manifest.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        sample_dir = self.gen_dir / sample.sample_id

        result = {
            "sample_id": sample.sample_id,
            "identity_id": sample.identity_id,
        }

        # Load generated frames
        gen_frames_dir = sample_dir / "gen_frames"
        if gen_frames_dir.exists():
            result["gen_frames"] = self._load_frames_from_dir(gen_frames_dir)

        # Load generated video
        gen_video_path = sample_dir / "gen.mp4"
        if gen_video_path.exists():
            result["gen_video_path"] = str(gen_video_path)

        # Load generation metadata
        meta_path = sample_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                result["gen_meta"] = json.load(f)

        return result

    def _load_frames_from_dir(self, frames_dir: Path) -> torch.Tensor:
        """Load frames from a directory of images."""
        frame_files = sorted(frames_dir.glob("*.png"))
        if not frame_files:
            frame_files = sorted(frames_dir.glob("*.jpg"))

        frames = []
        for frame_path in frame_files[:self.n_frames]:
            img = Image.open(frame_path).convert("RGB")
            img = img.resize(self.resolution, Image.LANCZOS)
            img_np = np.array(img)
            img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
            frames.append(img_tensor)

        # Pad if needed
        while len(frames) < self.n_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, *self.resolution))

        return torch.stack(frames[:self.n_frames], dim=0)
