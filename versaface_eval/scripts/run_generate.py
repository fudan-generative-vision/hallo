#!/usr/bin/env python3
"""
VersaFace Generation CLI

Generates talking face videos from evaluation manifests using the VersaFace model.

Output structure:
    {out_dir}/{sample_id}/
        gen.mp4           - Generated video with audio
        gen_frames/       - Individual frames
        audio.wav         - Input audio (copied)
        meta.json         - Generation metadata

Usage:
    python run_generate.py \
        --manifest data/manifests/hdtf_test.jsonl \
        --ckpt path/to/versaface.pth \
        --out eval_outputs/versaface_v1/hdtf \
        --seed 42

"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from versaface_eval.datasets.manifest import EvalManifest, EvalSample


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate talking face videos from evaluation manifest"
    )

    # Required arguments
    parser.add_argument(
        "--manifest", "-m",
        type=str,
        required=True,
        help="Path to evaluation manifest JSONL file"
    )
    parser.add_argument(
        "--ckpt", "-c",
        type=str,
        required=True,
        help="Path to VersaFace model checkpoint"
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        required=True,
        help="Output directory for generated videos"
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to inference config YAML (optional)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation"
    )

    # Generation parameters
    parser.add_argument(
        "--n_frames",
        type=int,
        default=14,
        help="Number of frames to generate per clip"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Output video FPS"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Output resolution (width height)"
    )

    # Inference parameters
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=40,
        help="Number of diffusion inference steps"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--pose_weight",
        type=float,
        default=1.0,
        help="Weight for pose/motion control"
    )
    parser.add_argument(
        "--face_weight",
        type=float,
        default=1.0,
        help="Weight for face identity"
    )
    parser.add_argument(
        "--lip_weight",
        type=float,
        default=1.0,
        help="Weight for lip synchronization"
    )

    # Subset selection
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index in manifest (for distributed generation)"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index in manifest (exclusive)"
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific sample IDs to generate"
    )

    # Flags
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Save individual frames in addition to video"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples that already have outputs"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be generated without running"
    )

    return parser.parse_args()


def load_versaface_model(
    checkpoint_path: str,
    config_path: Optional[str],
    device: str,
) -> Any:
    """
    Load VersaFace model from checkpoint.

    Returns a model object with a generate() method.
    """
    # TODO: Implement actual VersaFace model loading
    # This is a placeholder that should be replaced with actual model loading

    print(f"Loading VersaFace model from {checkpoint_path}")

    # Placeholder model class
    class VersaFaceModelPlaceholder:
        def __init__(self, device):
            self.device = device

        def generate(
            self,
            ref_image: torch.Tensor,
            audio: torch.Tensor,
            n_frames: int,
            **kwargs,
        ) -> torch.Tensor:
            """
            Generate video frames.

            Args:
                ref_image: (C, H, W) reference image
                audio: Audio tensor
                n_frames: Number of frames to generate

            Returns:
                (T, C, H, W) generated frames
            """
            # Placeholder: return random frames
            H, W = ref_image.shape[1], ref_image.shape[2]
            return torch.rand(n_frames, 3, H, W)

    return VersaFaceModelPlaceholder(device)


def generate_sample(
    model: Any,
    sample: EvalSample,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Generate video for a single sample.

    Returns metadata about the generation.
    """
    from PIL import Image
    import torchaudio

    start_time = time.time()

    # Load reference image
    ref_image = Image.open(sample.ref_frame_path).convert("RGB")
    ref_image = ref_image.resize(tuple(args.resolution), Image.LANCZOS)
    ref_tensor = torch.from_numpy(np.array(ref_image)).float().permute(2, 0, 1) / 255.0

    # Load audio
    audio, sr = torchaudio.load(sample.audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)

    # Generate
    torch.manual_seed(args.seed)
    gen_frames = model.generate(
        ref_image=ref_tensor.to(args.device),
        audio=audio.to(args.device),
        n_frames=args.n_frames,
        inference_steps=args.inference_steps,
        cfg_scale=args.cfg_scale,
        pose_weight=args.pose_weight,
        face_weight=args.face_weight,
        lip_weight=args.lip_weight,
    )

    generation_time = time.time() - start_time

    return {
        "gen_frames": gen_frames.cpu(),
        "generation_time": generation_time,
        "n_frames": gen_frames.shape[0],
    }


def save_output(
    gen_frames: torch.Tensor,
    sample: EvalSample,
    out_dir: Path,
    args: argparse.Namespace,
    gen_meta: Dict[str, Any],
) -> None:
    """Save generated output to disk."""
    from PIL import Image

    sample_dir = out_dir / sample.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save frames
    if args.save_frames:
        frames_dir = sample_dir / "gen_frames"
        frames_dir.mkdir(exist_ok=True)

        for i, frame in enumerate(gen_frames):
            frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(frame_np)
            img.save(frames_dir / f"{i:06d}.png")

    # Save video with audio
    try:
        # Import video saving utility
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from hallo.utils.util import tensor_to_video

        video_path = str(sample_dir / "gen.mp4")
        # Reshape for tensor_to_video: expects (C, T, H, W)
        video_tensor = gen_frames.permute(1, 0, 2, 3)
        tensor_to_video(video_tensor, video_path, sample.audio_path, fps=int(args.fps))
    except Exception as e:
        print(f"Warning: Could not save video with audio: {e}")
        # Fallback: save video without audio
        _save_video_fallback(gen_frames, sample_dir / "gen.mp4", args.fps)

    # Copy input audio
    shutil.copy2(sample.audio_path, sample_dir / "audio.wav")

    # Save metadata
    meta = {
        "sample_id": sample.sample_id,
        "identity_id": sample.identity_id,
        "dataset": sample.dataset,
        "ref_frame_path": sample.ref_frame_path,
        "audio_path": sample.audio_path,
        "gt_video_path": sample.gt_video_path,
        "generation": {
            "checkpoint": args.ckpt,
            "seed": args.seed,
            "inference_steps": args.inference_steps,
            "cfg_scale": args.cfg_scale,
            "pose_weight": args.pose_weight,
            "face_weight": args.face_weight,
            "lip_weight": args.lip_weight,
            "resolution": args.resolution,
            "n_frames": args.n_frames,
            "fps": args.fps,
        },
        "runtime": {
            "generation_time_sec": gen_meta["generation_time"],
            "timestamp": datetime.now().isoformat(),
            "device": args.device,
        },
    }

    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _save_video_fallback(
    frames: torch.Tensor,
    output_path: Path,
    fps: float,
) -> None:
    """Fallback video saving without audio."""
    try:
        import av

        container = av.open(str(output_path), mode="w")
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = frames.shape[3]
        stream.height = frames.shape[2]
        stream.pix_fmt = "yuv420p"

        for frame in frames:
            frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            av_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()
    except ImportError:
        print("Warning: pyav not installed, cannot save video")


def main():
    args = setup_args()

    # Load manifest
    print(f"Loading manifest from {args.manifest}")
    manifest = EvalManifest.load(args.manifest)
    print(f"Loaded {len(manifest)} samples from {manifest.dataset_name}")

    # Filter samples
    samples = list(manifest.samples)

    if args.sample_ids:
        sample_ids_set = set(args.sample_ids)
        samples = [s for s in samples if s.sample_id in sample_ids_set]
        print(f"Filtered to {len(samples)} samples by ID")

    if args.end_idx is not None:
        samples = samples[args.start_idx:args.end_idx]
    else:
        samples = samples[args.start_idx:]

    print(f"Will generate {len(samples)} samples")

    # Setup output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dry run check
    if args.dry_run:
        print("\n=== DRY RUN ===")
        for sample in samples[:10]:
            print(f"  {sample.sample_id}: {sample.ref_frame_path}")
        if len(samples) > 10:
            print(f"  ... and {len(samples) - 10} more")
        return

    # Load model
    model = load_versaface_model(args.ckpt, args.config, args.device)

    # Generate
    print(f"\nGenerating to {out_dir}")
    total_time = 0

    for i, sample in enumerate(samples):
        sample_out_dir = out_dir / sample.sample_id

        # Skip if exists
        if args.skip_existing and (sample_out_dir / "gen.mp4").exists():
            print(f"[{i+1}/{len(samples)}] Skipping {sample.sample_id} (exists)")
            continue

        print(f"[{i+1}/{len(samples)}] Generating {sample.sample_id}...")

        try:
            # Validate sample paths
            errors = sample.validate()
            if errors:
                print(f"  Warning: {errors}")
                continue

            # Generate
            result = generate_sample(model, sample, args)

            # Save
            save_output(
                gen_frames=result["gen_frames"],
                sample=sample,
                out_dir=out_dir,
                args=args,
                gen_meta=result,
            )

            total_time += result["generation_time"]
            print(f"  Done in {result['generation_time']:.2f}s")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Summary
    print(f"\n=== Generation Complete ===")
    print(f"Total samples: {len(samples)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per sample: {total_time / len(samples):.2f}s")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
