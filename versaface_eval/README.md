# VersaFace Evaluation Framework

Evaluation harness for VersaFace talking-face generation, following the Hallo evaluation protocol with extensions for motion/expression alignment.

## Overview

This framework provides:
- **Canonical manifest schema** for standardized evaluation samples
- **Identity-based dataset splitting** (90/10 train/test by identity)
- **Metric calculators**: FID, FVD, Sync-C, Sync-D, E-FID
- **Generation CLI**: Produce standardized outputs from VersaFace model
- **Evaluation CLI**: Compute metrics and generate paper-ready tables
- **Stress slice evaluation**: Silent, non-English, async segments

## Quick Start

### 1. Create Evaluation Manifest

```python
from versaface_eval.datasets import EvalManifest, EvalSample, create_identity_split

# Create manifest from your dataset
manifest = EvalManifest(dataset_name="hdtf")

for video_path, identity, ref_frame, audio in your_data:
    sample = EvalSample(
        sample_id=f"hdtf_{idx:06d}",
        identity_id=identity,
        dataset="hdtf",
        ref_frame_path=ref_frame,
        audio_path=audio,
        gt_video_path=video_path,
        fps=25.0,
        n_frames=14,
    )
    manifest.add_sample(sample)

# Save manifest
manifest.save("data/manifests/hdtf_test.jsonl")
```

### 2. Generate Outputs

```bash
python versaface_eval/scripts/run_generate.py \
    --manifest data/manifests/hdtf_test.jsonl \
    --ckpt path/to/versaface.pth \
    --out eval_outputs/versaface_v1/hdtf \
    --seed 42 \
    --save_frames
```

### 3. Evaluate

```bash
python versaface_eval/scripts/run_eval.py \
    --manifest data/manifests/hdtf_test.jsonl \
    --gen_dir eval_outputs/versaface_v1/hdtf \
    --out_metrics results/hdtf_metrics.json \
    --metrics fid fvd sync efid \
    --eval_slices \
    --latex
```

## Directory Structure

```
versaface_eval/
├── configs/
│   └── eval_config.yaml      # Default evaluation config
├── datasets/
│   ├── __init__.py
│   ├── manifest.py           # EvalSample, EvalManifest classes
│   ├── splits.py             # Identity-based splitting
│   └── loader.py             # PyTorch Dataset loaders
├── metrics/
│   ├── __init__.py
│   ├── base.py               # MetricCalculator base class
│   ├── fid.py                # Frame-level FID
│   ├── fvd.py                # Video-level FVD
│   ├── sync.py               # SyncNet lip-sync metrics
│   └── efid.py               # Expression FID
├── scripts/
│   ├── run_generate.py       # Generation CLI
│   └── run_eval.py           # Evaluation CLI
└── docs/
    └── hallo_eval_extraction_report.md
```

## Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **FID** | Frame-level Fréchet Inception Distance | ↓ Lower is better |
| **FVD** | Fréchet Video Distance (I3D features) | ↓ Lower is better |
| **Sync-C** | Lip sync confidence (cosine similarity) | ↑ Higher is better |
| **Sync-D** | Lip sync distance (L2 distance) | ↓ Lower is better |
| **E-FID** | Expression FID (3DMM parameters) | ↓ Lower is better |

## Manifest Format

Each sample in the manifest JSONL contains:

```json
{
  "sample_id": "hdtf_000001",
  "identity_id": "RD_Radio1",
  "dataset": "hdtf",
  "ref_frame_path": "/data/hdtf/frames/RD_Radio1/0001.png",
  "audio_path": "/data/hdtf/audio/RD_Radio1_001.wav",
  "gt_video_path": "/data/hdtf/videos/RD_Radio1_001.mp4",
  "fps": 25.0,
  "start_time": 0.0,
  "end_time": 0.56,
  "n_frames": 14,
  "is_silent": false,
  "is_non_english": false,
  "language": "en"
}
```

## Output Structure

Generated outputs follow this structure:

```
eval_outputs/{run_name}/{dataset}/{sample_id}/
├── gen.mp4           # Generated video with audio
├── gen_frames/       # Individual frames (optional)
│   ├── 000000.png
│   └── ...
├── audio.wav         # Input audio (copied)
└── meta.json         # Generation metadata
```

## Hallo Protocol Compliance

This framework adopts Hallo's evaluation conventions:
- **Resolution**: 512×512 (standard), 256×256 (dev)
- **Temporal**: 14 frames @ 25 fps (0.56s clips)
- **Face processing**: 1.2× expansion ratio
- **Identity split**: 90% train / 10% test by identity

Extensions for VersaFace:
- E-FID for expression distributional similarity
- Stress slices (silent, non-English, async)
- Per-sample metric tracking

## Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy scipy pillow av
pip install pytorch-fvd  # For FVD computation
```

Optional (for advanced metrics):
```bash
pip install clean-fid    # Alternative FID implementation
pip install emoca        # For E-FID with EMOCA
```

## Citation

If using this evaluation framework, please cite:

```bibtex
@article{versaface2024,
  title={VersaFace: ...},
  author={...},
  year={2024}
}

@article{hallo2024,
  title={Hallo: Hierarchical Audio-Driven Visual Synthesis for Portrait Image Animation},
  author={...},
  journal={arXiv preprint arXiv:2406.08801},
  year={2024}
}
```
