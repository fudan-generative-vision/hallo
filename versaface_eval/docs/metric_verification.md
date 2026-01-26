# Metric Verification Document

**Date**: 2026-01-26
**Status**: Verified against primary sources

This document records the verified definitions of each evaluation metric, with sources.

---

## 1. FID (Fréchet Inception Distance)

### Definition
Measures distribution similarity between generated and real images using features from InceptionV3.

### Computation
```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2))
```
Where:
- μ_r, Σ_r = mean and covariance of real image features
- μ_g, Σ_g = mean and covariance of generated image features
- Features extracted from InceptionV3 pool3 layer (2048-dim)

### Direction
**Lower is better** (0 = identical distributions)

### Sources
- [FID Wikipedia](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)
- [PyTorch-Metrics FID](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html)

### Implementation Status
✅ **VERIFIED** - Our implementation matches standard approach

---

## 2. FVD (Fréchet Video Distance)

### Definition
Video-level analog of FID, measuring distribution similarity using spatiotemporal features from I3D network.

### Computation
Same Fréchet distance formula as FID, but using I3D features instead of Inception features.

### Key Details
- Uses **I3D pretrained on Kinetics-400**
- Features from top pooling layer (typically 400-dim or 2048-dim depending on layer)
- Requires **fixed clip length** for all videos
- Input format: (N, C, T, H, W) or (N, T, C, H, W)

### Direction
**Lower is better**

### Sources
- [FVD Paper (ICLR 2019)](https://openreview.net/pdf?id=rylgEULtdN)
- [PyTorch-Frechet-Video-Distance](https://github.com/ragor114/PyTorch-Frechet-Video-Distance)
- [Google Research FVD](https://github.com/google-research/google-research/tree/master/frechet_video_distance)

### Implementation Recommendations
Use one of these verified implementations:
1. `ragor114/PyTorch-Frechet-Video-Distance` - Pure PyTorch
2. `cd-fvd` (PyPI) - CVPR 2024, addresses content bias
3. `JunyaoHu/common_metrics_on_video_quality` - Includes FVD, PSNR, SSIM, LPIPS

### ⚠️ Known Issues
- FVD sensitive to content bias from I3D training data
- "Frozen" videos (single frame repeated) may get misleadingly low FVD
- Recommend using alongside other temporal metrics

---

## 3. Sync-C and Sync-D (LSE-C and LSE-D)

### CRITICAL FINDING
**Sync-C = LSE-C (Lip Sync Error - Confidence)**
**Sync-D = LSE-D (Lip Sync Error - Distance)**

These are the same metrics with different names across papers.

### Definition

**LSE-D (Sync-D)**: Average distance between audio and lip embeddings from SyncNet
- Measures the L2 distance in SyncNet embedding space
- **Lower is better** (closer embeddings = better sync)
- Typical good values: 6-8 (Wav2Lip achieves ~6.4 on LRS2)

**LSE-C (Sync-C)**: Confidence score from SyncNet
- Measures difference between minimum and median distances across offsets
- **Higher is better** (more confident sync detection)
- Typical good values: 5-8

### Computation (from SyncNet)
```
# SyncNet outputs for a video:
# AV offset: 0, Min dist: 8.860, Confidence: 5.738
#                   ↑                    ↑
#               LSE-D (Sync-D)      LSE-C (Sync-C)
```

### Official Implementation
**USE THIS**: [joonson/syncnet_python](https://github.com/joonson/syncnet_python)

Pipeline:
```bash
python run_pipeline.py --videofile video.mp4 --reference name --data_dir /output
python run_syncnet.py --data_dir /output
```

### Sources
- [Wav2Lip Paper (ACM MM 2020)](https://arxiv.org/pdf/2008.10010)
- [Wav2Lip GitHub Issue #284](https://github.com/Rudrabha/Wav2Lip/issues/284) - Confirms LSE-D = Min dist, LSE-C = Confidence
- [syncnet_python](https://github.com/joonson/syncnet_python)

### ⚠️ Critical Limitations (from THEval paper)
**SyncNet metrics have POOR correlation with human evaluation:**
- LSE-C correlation with human preference: ρ = -0.164
- LSE-D correlation with human preference: ρ = -0.269

This means these metrics may not reflect actual perceptual quality!

### Implementation Status
⚠️ **NEEDS UPDATE** - Should wrap syncnet_python instead of custom implementation

---

## 4. E-FID (Expression FID)

### Definition
FID computed on facial expression features rather than Inception features. Measures whether the distribution of expressions in generated videos matches the ground truth.

### Verified Information
From [MF-ETalk paper](https://www.mdpi.com/2079-9292/14/13/2684):
> "Expression-FID (E-FID) measures expression semantic fidelity by quantifying the distribution difference of facial expression features between generated and real videos"

### Typical Values
- MF-ETalk on MEAD: E-FID = 2.403
- MF-ETalk on HDTF: E-FID = 3.127

### Computation Options

**Option 1: 3DMM Expression Parameters**
- Extract FLAME/3DMM expression coefficients (50-dim) + jaw pose (3-dim)
- Compute FID on these 53-dim vectors
- Models: EMOCA, DECA, FLAME

**Option 2: Expression Recognition Features**
- Use expression recognition network features
- Compute FID on these features

### Direction
**Lower is better**

### ⚠️ Remaining Uncertainty
The exact E-FID implementation varies across papers. Hallo paper does not specify their exact computation method.

**Our recommendation**: Use 3DMM expression parameters (EMOCA/DECA) as this is most commonly referenced.

### Implementation Status
⚠️ **UNCERTAIN** - Multiple valid approaches, we chose 3DMM-based

---

## 5. Dataset Verification

### HDTF (High-Definition Talking Face)

| Property | Value | Source |
|----------|-------|--------|
| Resolution | 720p/1080p original, 512×512 cropped | [HDTF GitHub](https://github.com/MRzzm/HDTF) |
| Total clips | ~300-400 (varies by filtering) | Hallo paper |
| Frame rate | 25 fps | Hallo code |
| License | CC BY 4.0 | HDTF README |

**Download**: [GitHub](https://github.com/MRzzm/HDTF), [HuggingFace](https://huggingface.co/datasets/global-optima-research/HDTF)

### CelebV-HQ

| Property | Value | Source |
|----------|-------|--------|
| Clips | 35,666 | [CelebV-HQ](https://celebv-hq.github.io/) |
| Identities | 15,653 | CelebV-HQ paper |
| Resolution | 512×512 minimum | CelebV-HQ paper |
| Duration | 3-20 seconds per clip | CelebV-HQ paper |
| License | Non-commercial only | CelebV-HQ GitHub |

**Download**: [GitHub](https://github.com/CelebV-HQ/CelebV-HQ)

### Wild Dataset
Hallo uses internet-sourced videos:
- 1617 clips raw → 1324 clips filtered
- 137.49 hours raw → 93.73 hours filtered
- No public release of this dataset

---

## 6. Hallo Evaluation Protocol Summary

From paper analysis:

| Parameter | Value |
|-----------|-------|
| Resolution | 512×512 (primary) |
| Clip length | 14-15 frames |
| Train/Test split | 90% / 10% by identity |
| Metrics | FID, FVD, Sync-C, Sync-D, E-FID |

### Baselines Compared
- SadTalker
- Audio2Head
- DreamTalk
- AniPortrait

---

## 7. Confidence Summary

| Metric | Confidence | Notes |
|--------|------------|-------|
| FID | ✅ High | Standard implementation |
| FVD | ✅ High | Use verified PyTorch implementation |
| Sync-C/Sync-D | ⚠️ Medium | Use syncnet_python; note poor human correlation |
| E-FID | ⚠️ Low | Multiple valid approaches; paper doesn't specify |

---

## 8. Recommended Actions

1. **Sync metrics**: Integrate `joonson/syncnet_python` as the canonical implementation
2. **FVD**: Use `ragor114/PyTorch-Frechet-Video-Distance` or `cd-fvd`
3. **E-FID**: Document our choice (3DMM-based) explicitly; note this is our interpretation
4. **Report THEval metrics**: Consider adding THEval as supplementary (better human correlation)
