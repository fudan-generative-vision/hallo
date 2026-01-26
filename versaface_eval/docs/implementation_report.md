# VersaFace Evaluation Framework - Implementation Report

## Overview

This report documents the implementation of the VersaFace evaluation framework, which replicates and extends the Hallo evaluation protocol for talking-face generation.

## Objectives

1. Replicate Hallo's evaluation protocol (datasets, splits, metrics)
2. Implement 5 metrics: FID, FVD, Sync-C, Sync-D, E-FID
3. Support 3 datasets: HDTF, CelebV, Wild
4. Create Generation CLI and Evaluation CLI
5. Document all metric definitions with verified sources

## Implementation Summary

### Files Created

```
versaface_eval/
├── __init__.py
├── configs/
│   └── eval_config.yaml
├── datasets/
│   ├── __init__.py
│   ├── manifest.py          # EvalSample, EvalManifest dataclasses
│   ├── splits.py            # Identity-disjoint 90/10 splits
│   └── loader.py            # Dataset loading utilities
├── metrics/
│   ├── __init__.py
│   ├── base.py              # MetricCalculator base class
│   ├── fid.py               # Fréchet Inception Distance
│   ├── fvd.py               # Fréchet Video Distance
│   ├── sync.py              # SyncNet metrics (Sync-C, Sync-D)
│   └── efid.py              # Expression FID
├── scripts/
│   ├── run_generate.py      # Generation CLI
│   └── run_eval.py          # Evaluation CLI
└── docs/
    ├── hallo_eval_extraction_report.md
    ├── metric_verification.md
    └── implementation_report.md (this file)
```

### Metrics Implemented

| Metric | Definition | Direction | Verified Source |
|--------|------------|-----------|-----------------|
| **FID** | Fréchet distance in InceptionV3 pool3 space (2048-dim) | Lower is better | PyTorch-Metrics, clean-fid |
| **FVD** | Fréchet distance in I3D feature space (Kinetics-400) | Lower is better | ICLR 2019, Google Research |
| **Sync-C** | SyncNet confidence score (= LSE-C from Wav2Lip) | Higher is better | Wav2Lip (ACM MM 2020) |
| **Sync-D** | SyncNet min distance (= LSE-D from Wav2Lip) | Lower is better | Wav2Lip (ACM MM 2020) |
| **E-FID** | Fréchet distance in 3DMM expression space (53-dim) | Lower is better | MF-ETalk (MDPI 2024) |

### Key Findings from Verification

#### 1. SyncNet Metrics (Sync-C, Sync-D)

**Verified definitions:**
- Sync-C = LSE-C = "Confidence" from SyncNet
- Sync-D = LSE-D = "Min dist" from SyncNet

**Critical finding:** SyncNet metrics have **poor correlation with human evaluation** (ρ ≈ -0.2 per THEval paper). We documented this limitation and recommend supplementing with perceptual studies.

**Implementation:** Created `SyncNetWrapper` to integrate official joonson/syncnet_python for reproducibility.

#### 2. E-FID Definition

**Uncertainty:** E-FID is NOT standardized across papers. Hallo does not specify their exact computation.

**Options found:**
1. 3DMM expression parameters (50-dim + 3-dim jaw) - **our choice**
2. Expression recognition network features
3. Emotion embedding features

**Typical values:** MEAD ≈ 2.4, HDTF ≈ 3.1 (from MF-ETalk)

#### 3. FVD Known Issues

- Sensitive to content bias from I3D training data (Kinetics-400)
- "Frozen" videos may get misleadingly low FVD
- Recommended implementations: ragor114/PyTorch-Frechet-Video-Distance, cd-fvd

### Hallo Evaluation Protocol

From the Hallo paper (arXiv 2406.08801):

| Parameter | Value |
|-----------|-------|
| Resolution | 512 × 512 |
| Frame rate | 25 fps |
| Clip length | 14 frames |
| Face expansion | 1.2× |
| Test datasets | HDTF, CelebV |

### VersaFace Autoencoder Module

Additionally created `versaface_autoencoder/` package documenting Hallo's architecture:

```
versaface_autoencoder/
├── __init__.py
├── configs/
│   └── autoencoder_config.yaml
├── docs/
│   └── architecture.md
└── models/
    ├── __init__.py
    ├── vae_wrapper.py      # Diffusers AutoencoderKL wrapper
    ├── face_encoder.py     # InsightFace + ImageProjModel
    └── audio_encoder.py    # wav2vec2 + AudioProjModel
```

**Architecture highlights:**
- VAE: 8× spatial compression, 4 latent channels, 0.18215 scaling
- Face encoding: 512-dim InsightFace → 4×1024 context tokens
- Audio encoding: wav2vec2 (12×768) → 32×768 context tokens per frame
- UNet3D with motion modules and mutual self-attention

## Commits

1. `841c1d6` - docs: Add verified metric definitions and improve implementations
2. `74a1aa6` - feat: Add VersaFace autoencoder module with architecture documentation

## Remaining Work

1. **Integration testing:** End-to-end pipeline validation with sample data
2. **VersaFace model integration:** Replace placeholder in run_generate.py
3. **THEval metrics:** Consider adding as supplementary (better human correlation)
4. **Dataset preparation:** Scripts for HDTF/CelebV downloading and preprocessing

## References

- Hallo Paper: https://arxiv.org/abs/2406.08801
- Wav2Lip Paper: https://arxiv.org/abs/2008.10010
- SyncNet Official: https://github.com/joonson/syncnet_python
- THEval Paper: https://arxiv.org/abs/2311.17773
- MF-ETalk: https://www.mdpi.com/2079-9292/14/13/2684
- FVD Paper: https://openreview.net/pdf?id=rylgEULtdN

---

*Report generated: 2026-01-26*
*Branch: claude/understand-eval-framework-9MIN8*
