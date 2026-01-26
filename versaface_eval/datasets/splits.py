"""
Identity-based dataset splitting for evaluation.

Implements 90/10 train/test split by identity to ensure:
- No identity leakage between train and test
- Deterministic, reproducible splits
- Stored as persistent manifests
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class IdentitySplit:
    """Container for identity-based train/test split."""
    dataset_name: str
    train_identities: List[str]
    test_identities: List[str]
    seed: int
    train_ratio: float
    version: str = "1.0"

    @property
    def n_train(self) -> int:
        return len(self.train_identities)

    @property
    def n_test(self) -> int:
        return len(self.test_identities)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_test

    @property
    def actual_train_ratio(self) -> float:
        return self.n_train / self.n_total if self.n_total > 0 else 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "IdentitySplit":
        return cls(**data)

    def save(self, path: str) -> None:
        """Save split to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "IdentitySplit":
        """Load split from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


def create_identity_split(
    identities: List[str],
    dataset_name: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> IdentitySplit:
    """
    Create deterministic identity-based train/test split.

    Args:
        identities: List of unique identity IDs
        dataset_name: Name of the dataset
        train_ratio: Fraction of identities for training (default 0.9)
        seed: Random seed for reproducibility

    Returns:
        IdentitySplit object with train and test identity lists
    """
    # Ensure deterministic ordering
    identities = sorted(list(set(identities)))

    # Shuffle with fixed seed
    rng = random.Random(seed)
    shuffled = identities.copy()
    rng.shuffle(shuffled)

    # Split
    n_train = int(len(shuffled) * train_ratio)
    train_identities = sorted(shuffled[:n_train])
    test_identities = sorted(shuffled[n_train:])

    return IdentitySplit(
        dataset_name=dataset_name,
        train_identities=train_identities,
        test_identities=test_identities,
        seed=seed,
        train_ratio=train_ratio,
    )


def load_identity_split(path: str) -> IdentitySplit:
    """Load identity split from file."""
    return IdentitySplit.load(path)


def extract_identities_from_paths(
    video_paths: List[str],
    extraction_mode: str = "stem_prefix",
) -> Dict[str, str]:
    """
    Extract identity IDs from video paths.

    Args:
        video_paths: List of video file paths
        extraction_mode: How to extract identity
            - "stem_prefix": Use first part before underscore (e.g., "id001_clip1.mp4" -> "id001")
            - "parent": Use parent directory name
            - "stem": Use full stem

    Returns:
        Dict mapping video_path -> identity_id
    """
    path_to_identity = {}

    for path in video_paths:
        p = Path(path)
        if extraction_mode == "stem_prefix":
            # Common format: {identity}_{clip_number}.mp4
            identity = p.stem.split('_')[0]
        elif extraction_mode == "parent":
            # Format: {identity}/{clip}.mp4
            identity = p.parent.name
        elif extraction_mode == "stem":
            identity = p.stem
        else:
            raise ValueError(f"Unknown extraction_mode: {extraction_mode}")

        path_to_identity[path] = identity

    return path_to_identity


def get_identity_statistics(
    path_to_identity: Dict[str, str]
) -> Dict[str, int]:
    """
    Get sample counts per identity.

    Returns:
        Dict mapping identity_id -> number of samples
    """
    counts = {}
    for identity in path_to_identity.values():
        counts[identity] = counts.get(identity, 0) + 1
    return counts


def validate_split_disjoint(split: IdentitySplit) -> bool:
    """Verify train and test identities are disjoint."""
    train_set = set(split.train_identities)
    test_set = set(split.test_identities)
    return len(train_set & test_set) == 0


def create_split_from_video_list(
    video_paths: List[str],
    dataset_name: str,
    extraction_mode: str = "stem_prefix",
    train_ratio: float = 0.9,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> Tuple[IdentitySplit, Dict[str, str]]:
    """
    Convenience function to create split from video file list.

    Args:
        video_paths: List of video file paths
        dataset_name: Name of the dataset
        extraction_mode: How to extract identity from path
        train_ratio: Fraction for training
        seed: Random seed
        output_path: Optional path to save split JSON

    Returns:
        Tuple of (IdentitySplit, path_to_identity mapping)
    """
    # Extract identities
    path_to_identity = extract_identities_from_paths(video_paths, extraction_mode)

    # Get unique identities
    identities = list(set(path_to_identity.values()))

    # Create split
    split = create_identity_split(
        identities=identities,
        dataset_name=dataset_name,
        train_ratio=train_ratio,
        seed=seed,
    )

    # Optionally save
    if output_path:
        split.save(output_path)

    return split, path_to_identity


def filter_paths_by_split(
    video_paths: List[str],
    path_to_identity: Dict[str, str],
    split: IdentitySplit,
    subset: str = "test",
) -> List[str]:
    """
    Filter video paths to only include those in train or test split.

    Args:
        video_paths: List of video paths
        path_to_identity: Mapping from path to identity
        split: The identity split
        subset: "train" or "test"

    Returns:
        Filtered list of paths
    """
    if subset == "train":
        allowed_identities = set(split.train_identities)
    elif subset == "test":
        allowed_identities = set(split.test_identities)
    else:
        raise ValueError(f"subset must be 'train' or 'test', got '{subset}'")

    return [p for p in video_paths if path_to_identity.get(p) in allowed_identities]
