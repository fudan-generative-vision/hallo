"""
Canonical manifest schema for VersaFace evaluation.

Each test sample contains:
- Reference image: a still frame of the target identity
- Driving audio: a segment of speech
- Ground-truth video: the corresponding real clip (for distributional metrics)

Manifest format follows Hallo conventions with extensions for evaluation.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
import hashlib


@dataclass
class EvalSample:
    """
    Canonical evaluation sample.

    This is the unit of evaluation - one sample produces one generated clip
    that is compared against the ground truth for metrics.
    """
    # Identifiers
    sample_id: str                    # Unique sample identifier
    identity_id: str                  # Identity for train/test splitting
    dataset: str                      # Source dataset (hdtf, celebv, wild)

    # Input paths (for generation)
    ref_frame_path: str               # Reference image path
    audio_path: str                   # Driving audio path

    # Ground truth paths (for evaluation)
    gt_video_path: str                # Ground truth video path

    # Temporal specification
    fps: float = 25.0                 # Video FPS
    start_time: float = 0.0           # Clip start time (seconds)
    end_time: float = 0.0             # Clip end time (seconds)
    n_frames: int = 14                # Number of frames in clip

    # Preprocessing paths (optional, filled during preprocessing)
    face_mask_path: Optional[str] = None
    lip_mask_path: Optional[str] = None
    face_emb_path: Optional[str] = None
    audio_emb_path: Optional[str] = None

    # Stress slice annotations (optional)
    is_silent: bool = False           # Low VAD activity
    is_non_english: bool = False      # Non-English speech
    language: Optional[str] = None    # Detected language
    vad_score: Optional[float] = None # Voice activity score

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Clip duration in seconds."""
        return self.end_time - self.start_time

    @property
    def clip_id(self) -> str:
        """Generate deterministic clip ID from paths and timing."""
        content = f"{self.gt_video_path}:{self.start_time}:{self.end_time}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalSample":
        """Create from dictionary."""
        return cls(**data)

    def validate(self) -> List[str]:
        """Validate sample paths and return list of errors."""
        errors = []

        if not Path(self.ref_frame_path).exists():
            errors.append(f"Reference frame not found: {self.ref_frame_path}")
        if not Path(self.audio_path).exists():
            errors.append(f"Audio not found: {self.audio_path}")
        if not Path(self.gt_video_path).exists():
            errors.append(f"Ground truth video not found: {self.gt_video_path}")

        if self.end_time <= self.start_time:
            errors.append(f"Invalid time range: {self.start_time} to {self.end_time}")
        if self.n_frames <= 0:
            errors.append(f"Invalid frame count: {self.n_frames}")

        return errors


@dataclass
class EvalManifest:
    """
    Collection of evaluation samples for a dataset.

    Provides:
    - Load/save to JSONL format
    - Filtering by identity, stress slices
    - Statistics and validation
    """
    samples: List[EvalSample] = field(default_factory=list)
    dataset_name: str = ""
    version: str = "1.0"

    # Split information
    split: str = "test"  # train, val, test
    identity_split_path: Optional[str] = None

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self.samples[idx]

    def add_sample(self, sample: EvalSample) -> None:
        """Add a sample to the manifest."""
        self.samples.append(sample)

    def get_identities(self) -> List[str]:
        """Get unique identity IDs."""
        return list(set(s.identity_id for s in self.samples))

    def filter_by_identities(self, identity_ids: List[str]) -> "EvalManifest":
        """Return new manifest with only specified identities."""
        filtered = [s for s in self.samples if s.identity_id in identity_ids]
        return EvalManifest(
            samples=filtered,
            dataset_name=self.dataset_name,
            version=self.version,
            split=self.split,
            identity_split_path=self.identity_split_path,
        )

    def filter_silent(self) -> "EvalManifest":
        """Return manifest with only silent segments."""
        filtered = [s for s in self.samples if s.is_silent]
        manifest = EvalManifest(
            samples=filtered,
            dataset_name=f"{self.dataset_name}_silent",
            version=self.version,
            split=self.split,
        )
        return manifest

    def filter_non_english(self) -> "EvalManifest":
        """Return manifest with only non-English segments."""
        filtered = [s for s in self.samples if s.is_non_english]
        manifest = EvalManifest(
            samples=filtered,
            dataset_name=f"{self.dataset_name}_non_english",
            version=self.version,
            split=self.split,
        )
        return manifest

    def get_statistics(self) -> Dict[str, Any]:
        """Get manifest statistics."""
        durations = [s.duration for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "n_identities": len(self.get_identities()),
            "total_duration_sec": sum(durations),
            "mean_duration_sec": sum(durations) / len(durations) if durations else 0,
            "n_silent": sum(1 for s in self.samples if s.is_silent),
            "n_non_english": sum(1 for s in self.samples if s.is_non_english),
            "datasets": list(set(s.dataset for s in self.samples)),
        }

    def validate(self) -> Dict[str, List[str]]:
        """Validate all samples and return errors by sample_id."""
        errors = {}
        for sample in self.samples:
            sample_errors = sample.validate()
            if sample_errors:
                errors[sample.sample_id] = sample_errors
        return errors

    def save(self, path: str) -> None:
        """Save manifest to JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            # Write header as first line
            header = {
                "_type": "manifest_header",
                "dataset_name": self.dataset_name,
                "version": self.version,
                "split": self.split,
                "identity_split_path": self.identity_split_path,
                "n_samples": len(self.samples),
            }
            f.write(json.dumps(header) + '\n')

            # Write samples
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict()) + '\n')

    @classmethod
    def load(cls, path: str) -> "EvalManifest":
        """Load manifest from JSONL file."""
        samples = []
        header = {}

        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if data.get("_type") == "manifest_header":
                    header = data
                else:
                    samples.append(EvalSample.from_dict(data))

        return cls(
            samples=samples,
            dataset_name=header.get("dataset_name", ""),
            version=header.get("version", "1.0"),
            split=header.get("split", "test"),
            identity_split_path=header.get("identity_split_path"),
        )

    @classmethod
    def from_hallo_metadata(cls, metadata_path: str, dataset_name: str) -> "EvalManifest":
        """
        Create manifest from Hallo-style metadata JSON.

        Hallo metadata format:
        {
            "video_path": "...",
            "mask_path": "...",
            "face_emb_path": "...",
            "audio_path": "...",
            ...
        }
        """
        with open(metadata_path, 'r') as f:
            # Hallo uses JSON array or JSON lines
            content = f.read().strip()
            if content.startswith('['):
                items = json.loads(content)
            else:
                items = [json.loads(line) for line in content.split('\n') if line.strip()]

        samples = []
        for idx, item in enumerate(items):
            video_path = item.get("video_path", "")
            # Extract identity from video path (assumes path contains identity info)
            identity_id = Path(video_path).stem.split('_')[0] if video_path else f"id_{idx}"

            sample = EvalSample(
                sample_id=f"{dataset_name}_{idx:06d}",
                identity_id=identity_id,
                dataset=dataset_name,
                ref_frame_path=item.get("ref_frame_path", ""),  # May need extraction
                audio_path=item.get("audio_path", ""),
                gt_video_path=video_path,
                face_mask_path=item.get("mask_path"),
                lip_mask_path=item.get("sep_mask_lip"),
                face_emb_path=item.get("face_emb_path"),
                audio_emb_path=item.get("vocals_emb_base_all"),
            )
            samples.append(sample)

        return cls(samples=samples, dataset_name=dataset_name)
