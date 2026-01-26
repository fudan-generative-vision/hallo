#!/usr/bin/env python3
"""
VersaFace Evaluation CLI

Computes evaluation metrics on generated outputs.

Metrics computed:
- FID (Frame-level Fréchet Inception Distance)
- FVD (Fréchet Video Distance)
- Sync-C (Lip sync confidence)
- Sync-D (Lip sync distance)
- E-FID (Expression FID)

Usage:
    python run_eval.py \
        --manifest data/manifests/hdtf_test.jsonl \
        --gen_dir eval_outputs/versaface_v1/hdtf \
        --out_metrics results/hdtf_metrics.json

"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from versaface_eval.datasets.manifest import EvalManifest
from versaface_eval.datasets.loader import EvalDataset, GeneratedDataset
from versaface_eval.metrics import (
    FIDCalculator,
    FVDCalculator,
    SyncNetCalculator,
    EFIDCalculator,
)
from versaface_eval.metrics.base import MetricResult


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated talking face videos"
    )

    # Required arguments
    parser.add_argument(
        "--manifest", "-m",
        type=str,
        required=True,
        help="Path to evaluation manifest JSONL file"
    )
    parser.add_argument(
        "--gen_dir", "-g",
        type=str,
        required=True,
        help="Directory containing generated outputs"
    )
    parser.add_argument(
        "--out_metrics", "-o",
        type=str,
        required=True,
        help="Output path for metrics JSON"
    )

    # Metric selection
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["fid", "fvd", "sync", "efid"],
        choices=["fid", "fvd", "sync", "efid", "all"],
        help="Metrics to compute"
    )

    # Optional arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for metric computation"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=14,
        help="Number of frames per clip"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Video resolution (width height)"
    )

    # Stress slice evaluation
    parser.add_argument(
        "--eval_slices",
        action="store_true",
        help="Also evaluate on stress slices (silent, non-english)"
    )

    # SyncNet specific
    parser.add_argument(
        "--syncnet_checkpoint",
        type=str,
        default=None,
        help="Path to SyncNet checkpoint"
    )

    # Subset selection
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )

    # Output options
    parser.add_argument(
        "--save_per_sample",
        action="store_true",
        help="Save per-sample metric scores"
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also output LaTeX table row"
    )

    return parser.parse_args()


def load_video_frames(video_path: str, n_frames: int, resolution: tuple) -> torch.Tensor:
    """Load frames from video file."""
    try:
        import av
        from PIL import Image

        container = av.open(video_path)
        stream = container.streams.video[0]

        frames = []
        for frame in container.decode(video=0):
            img = frame.to_image().convert("RGB")
            img = img.resize(resolution, Image.LANCZOS)
            img_np = np.array(img)
            img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
            frames.append(img_tensor)

            if len(frames) >= n_frames:
                break

        container.close()

        # Pad if needed
        while len(frames) < n_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, *resolution[::-1]))

        return torch.stack(frames[:n_frames])

    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None


def load_frames_from_dir(frames_dir: str, n_frames: int, resolution: tuple) -> torch.Tensor:
    """Load frames from directory of images."""
    from PIL import Image

    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.jpg"))

    frames = []
    for frame_path in frame_files[:n_frames]:
        img = Image.open(frame_path).convert("RGB")
        img = img.resize(resolution, Image.LANCZOS)
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
        frames.append(img_tensor)

    # Pad if needed
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else torch.zeros(3, *resolution[::-1]))

    return torch.stack(frames[:n_frames])


def evaluate_metrics(
    manifest: EvalManifest,
    gen_dir: Path,
    args: argparse.Namespace,
    slice_name: str = "full",
) -> Dict[str, MetricResult]:
    """
    Evaluate all requested metrics on a manifest.

    Returns dict of metric_name -> MetricResult
    """
    results = {}
    resolution = tuple(args.resolution)

    # Determine which metrics to compute
    metrics_to_compute = args.metrics
    if "all" in metrics_to_compute:
        metrics_to_compute = ["fid", "fvd", "sync", "efid"]

    # Initialize calculators
    calculators = {}

    if "fid" in metrics_to_compute:
        calculators["fid"] = FIDCalculator(
            device=args.device,
            batch_size=args.batch_size,
        )

    if "fvd" in metrics_to_compute:
        calculators["fvd"] = FVDCalculator(
            device=args.device,
            batch_size=args.batch_size,
            n_frames=args.n_frames,
        )

    if "sync" in metrics_to_compute:
        calculators["sync"] = SyncNetCalculator(
            device=args.device,
            syncnet_checkpoint=args.syncnet_checkpoint,
        )

    if "efid" in metrics_to_compute:
        calculators["efid"] = EFIDCalculator(
            device=args.device,
            batch_size=args.batch_size,
        )

    # Limit samples if requested
    samples = list(manifest.samples)
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"\nEvaluating {len(samples)} samples for slice: {slice_name}")
    print(f"Metrics: {list(calculators.keys())}")

    # Process samples
    n_processed = 0
    n_skipped = 0

    for i, sample in enumerate(samples):
        sample_dir = gen_dir / sample.sample_id

        # Check if generated output exists
        gen_video_path = sample_dir / "gen.mp4"
        gen_frames_dir = sample_dir / "gen_frames"

        if not gen_video_path.exists() and not gen_frames_dir.exists():
            n_skipped += 1
            continue

        # Load generated frames
        if gen_frames_dir.exists():
            gen_frames = load_frames_from_dir(gen_frames_dir, args.n_frames, resolution)
        else:
            gen_frames = load_video_frames(str(gen_video_path), args.n_frames, resolution)

        if gen_frames is None:
            n_skipped += 1
            continue

        # Load ground truth frames
        gt_frames = load_video_frames(sample.gt_video_path, args.n_frames, resolution)
        if gt_frames is None:
            n_skipped += 1
            continue

        # Add batch dimension
        gen_frames = gen_frames.unsqueeze(0)  # (1, T, C, H, W)
        gt_frames = gt_frames.unsqueeze(0)

        # Update calculators
        if "fid" in calculators:
            calculators["fid"].update(gen_frames=gen_frames, gt_frames=gt_frames)

        if "fvd" in calculators:
            calculators["fvd"].update(gen_frames=gen_frames, gt_frames=gt_frames)

        if "efid" in calculators:
            calculators["efid"].update(gen_frames=gen_frames, gt_frames=gt_frames)

        # Sync metrics need audio
        if "sync" in calculators and os.path.exists(sample.audio_path):
            try:
                import torchaudio
                audio, sr = torchaudio.load(sample.audio_path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    audio = resampler(audio)
                audio = audio.unsqueeze(0)  # Add batch dim
                calculators["sync"].update(gen_frames=gen_frames, audio=audio)
            except Exception as e:
                print(f"  Warning: Could not process audio for sync: {e}")

        n_processed += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")

    print(f"  Processed: {n_processed}, Skipped: {n_skipped}")

    # Compute final metrics
    for name, calculator in calculators.items():
        try:
            result = calculator.compute()
            results[name] = result
            print(f"  {result}")
        except Exception as e:
            print(f"  Error computing {name}: {e}")

    return results


def format_latex_row(
    results: Dict[str, MetricResult],
    dataset_name: str,
    model_name: str,
) -> str:
    """Format results as LaTeX table row."""
    # Expected columns: Model & Dataset & FID↓ & FVD↓ & Sync-C↑ & Sync-D↓ & E-FID↓

    fid = results.get("fid")
    fvd = results.get("fvd")
    sync = results.get("sync")
    efid = results.get("efid")

    fid_val = f"{fid.value:.2f}" if fid else "-"
    fvd_val = f"{fvd.value:.2f}" if fvd else "-"
    sync_c_val = f"{sync.details['sync_c']:.3f}" if sync else "-"
    sync_d_val = f"{sync.details['sync_d']:.3f}" if sync else "-"
    efid_val = f"{efid.value:.2f}" if efid else "-"

    return f"{model_name} & {dataset_name} & {fid_val} & {fvd_val} & {sync_c_val} & {sync_d_val} & {efid_val} \\\\"


def main():
    args = setup_args()

    # Load manifest
    print(f"Loading manifest from {args.manifest}")
    manifest = EvalManifest.load(args.manifest)
    print(f"Loaded {len(manifest)} samples from {manifest.dataset_name}")

    gen_dir = Path(args.gen_dir)
    if not gen_dir.exists():
        raise ValueError(f"Generated output directory not found: {gen_dir}")

    # Evaluate on full manifest
    start_time = time.time()
    results = evaluate_metrics(manifest, gen_dir, args, slice_name="full")
    eval_time = time.time() - start_time

    # Evaluate on stress slices if requested
    slice_results = {}
    if args.eval_slices:
        # Silent slice
        silent_manifest = manifest.filter_silent()
        if len(silent_manifest) > 0:
            print(f"\n=== Evaluating SILENT slice ({len(silent_manifest)} samples) ===")
            slice_results["silent"] = evaluate_metrics(
                silent_manifest, gen_dir, args, slice_name="silent"
            )

        # Non-English slice
        non_english_manifest = manifest.filter_non_english()
        if len(non_english_manifest) > 0:
            print(f"\n=== Evaluating NON-ENGLISH slice ({len(non_english_manifest)} samples) ===")
            slice_results["non_english"] = evaluate_metrics(
                non_english_manifest, gen_dir, args, slice_name="non_english"
            )

    # Prepare output
    output = {
        "dataset": manifest.dataset_name,
        "manifest_path": args.manifest,
        "gen_dir": str(gen_dir),
        "n_samples": len(manifest),
        "metrics": {name: result.to_dict() for name, result in results.items()},
        "slice_metrics": {
            slice_name: {name: result.to_dict() for name, result in slice_results_dict.items()}
            for slice_name, slice_results_dict in slice_results.items()
        },
        "config": {
            "n_frames": args.n_frames,
            "resolution": args.resolution,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "runtime": {
            "eval_time_sec": eval_time,
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Save metrics
    out_path = Path(args.out_metrics)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Evaluation Complete ===")
    print(f"Results saved to: {out_path}")

    # Print summary
    print(f"\n=== Results Summary ({manifest.dataset_name}) ===")
    for name, result in results.items():
        print(f"  {result}")

    # Print slice results
    for slice_name, slice_results_dict in slice_results.items():
        print(f"\n=== {slice_name.upper()} Slice ===")
        for name, result in slice_results_dict.items():
            print(f"  {result}")

    # Output LaTeX if requested
    if args.latex:
        print(f"\n=== LaTeX Table Row ===")
        latex_row = format_latex_row(results, manifest.dataset_name, "VersaFace")
        print(latex_row)

        # Also save to file
        latex_path = out_path.with_suffix(".tex")
        with open(latex_path, "w") as f:
            f.write("% Model & Dataset & FID↓ & FVD↓ & Sync-C↑ & Sync-D↓ & E-FID↓\n")
            f.write(latex_row + "\n")
        print(f"LaTeX saved to: {latex_path}")


if __name__ == "__main__":
    main()
