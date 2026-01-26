# Hallo Evaluation Extraction Report

**Date**: 2026-01-26
**Purpose**: Document what evaluation infrastructure exists in Hallo repo vs what must be implemented for VersaFace

---

## Executive Summary

The Hallo repository provides **data preprocessing** and **training pipelines** but contains **NO evaluation metrics infrastructure**. We must implement all quantitative evaluation (FID, FVD, Sync-C, Sync-D, E-FID) from scratch while adopting their data format conventions.

---

## 1. Canonical Clip Format (Resolution, Length, Cropping)

### Resolution
- **Standard**: 512×512 (fixed across all configs)
- **Source**: `configs/train/stage1.yaml:3-4`, `configs/train/stage2.yaml:4-5`, `configs/inference/default.yaml:9-10`
- **Recommendation**: Adopt 512×512 as primary; support 256×256 for dev iteration

### Temporal Settings
| Parameter | Value | Source |
|-----------|-------|--------|
| FPS | 25 fps | `hallo/utils/util.py:909` (hardcoded in ffmpeg command) |
| n_sample_frames | 14 frames | `configs/train/stage2.yaml:9` |
| n_motion_frames | 2 frames | `configs/train/stage2.yaml:8` |
| audio_margin | 2 frames | `configs/train/stage2.yaml:10` |
| Minimum video length | 20 frames | Asserted in `hallo/datasets/talk_video.py:218-221` |

**Recommendation**: Use 14-frame clips at 25 fps (0.56 seconds) as canonical evaluation unit

### Face Cropping/Alignment
- **Detection**: InsightFace with 640×640 detection size
- **Selection**: Largest face by bounding box area
- **Expansion**: `face_expand_ratio: 1.2` (default)
- **Masks generated**: face, lip, pose/background, blurred variants
- **Source**: `hallo/datasets/image_processor.py:100-136`, `hallo/utils/util.py:407-564`

---

## 2. Metric Implementations in Hallo Repo

### STATUS: **ALL MISSING**

| Metric | Present? | Notes |
|--------|----------|-------|
| FID (frame-level) | ❌ No | Must implement |
| FVD (video-level) | ❌ No | Must implement with I3D backbone |
| Sync-C (lip sync confidence) | ❌ No | Must implement with SyncNet |
| Sync-D (lip sync distance) | ❌ No | Must implement with SyncNet |
| E-FID (expression FID) | ❌ No | Must implement with 3DMM/expression encoder |
| LPIPS | ❌ No | Not in Hallo's reported metrics |
| SSIM/PSNR | ❌ No | Not in Hallo's reported metrics |

**Training loss only**: MSE with SNR weighting (`scripts/train_stage1.py:622-664`, `scripts/train_stage2.py:825-880`)

**Validation during training**: Visual only - saves comparison images (stage1) or videos (stage2) for manual inspection

---

## 3. Dataset & Split Conventions

### Datasets Referenced
- **HDTF**: Referenced in `configs/train/stage2.yaml:12` as `hdtf_split_stage2.json`
- **CelebV**: Not explicitly referenced in code
- **Wild dataset**: Not present

### Identity Split Logic
**STATUS: NOT IMPLEMENTED**

- Config references `hdtf_split_stage2.json` but **no script generates this file**
- No train/test/validation split code exists
- No identity-based deduplication logic
- All data loaded as training data without filtering

**Recommendation**: Implement 90/10 identity-disjoint split ourselves

### Metadata Schema (Stage 2)
From `scripts/extract_meta_info_stage2.py:142-151`:
```json
{
  "video_path": "path/to/video.mp4",
  "mask_path": "path/to/face_mask.png",
  "sep_mask_border": "path/to/sep_pose_mask.png",
  "sep_mask_face": "path/to/sep_face_mask.png",
  "sep_mask_lip": "path/to/sep_lip_mask.png",
  "face_emb_path": "path/to/face_emb.pt",
  "audio_path": "path/to/audio.wav",
  "vocals_emb_base_all": "path/to/audio_emb.pt"
}
```

---

## 4. Output Directory Structure

### Data Preprocessing Output
```
dataset_name/
├── videos/           # Original videos
├── images/           # Extracted frames at 25fps
│   └── {video_id}/
│       └── {frame:04d}.png
├── audios/           # Extracted audio (16kHz WAV)
├── face_mask/        # Binary face masks
├── sep_pose_mask/    # Background/pose masks
├── sep_face_mask/    # Face-only masks
├── sep_lip_mask/     # Lip-only masks
├── face_emb/         # InsightFace embeddings (.pt)
└── audio_emb/        # Wav2Vec2 embeddings (.pt)
```

### Training Output
```
exp_output/{exp_name}/
├── checkpoints/checkpoint-{step}/
├── modules/{module}-{step}.pth
├── validation/{step:06d}-{ref}_{mask}.jpg
└── config.yaml
```

### Inference Output
- Default: `.cache/output.mp4`
- Temp files: `{save_path}/audio_preprocess/`

---

## 5. Reusable Utilities from Hallo

### Can Adopt Directly
1. **Face mask generation**: `hallo/utils/util.py` functions:
   - `get_face_mask()`, `get_lip_mask()`, `get_union_mask()`
   - `blur_mask()`, `expand_region()`

2. **Video/audio utilities**: `hallo/utils/util.py`:
   - `tensor_to_video()` - converts tensor to MP4 with audio
   - `extract_audio_from_videos()` - ffmpeg wrapper
   - `convert_video_to_images()` - frame extraction at 25fps
   - `get_fps()`, `read_frames()` - video reading

3. **Face detection**: `hallo/datasets/image_processor.py`:
   - InsightFace-based face detection and embedding extraction

4. **Audio processing**: `hallo/datasets/audio_processor.py`:
   - Wav2Vec2 embedding extraction
   - Audio separation (vocals extraction)

### Must Implement
1. All evaluation metrics (FID, FVD, Sync-C, Sync-D, E-FID)
2. Identity-based dataset splitting
3. Canonical manifest format for evaluation
4. Generation CLI wrapper
5. Evaluation CLI wrapper
6. Stress slice creation (silent, non-English, async)

---

## 6. Adoption Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Resolution | 512×512 primary, 256×256 dev | Match Hallo standard |
| Clip length | 14 frames @ 25fps | Match Hallo n_sample_frames |
| Face expansion | 1.2× | Match Hallo default |
| FPS | 25 fps | Match Hallo hardcoded value |
| Audio sample rate | 16kHz | Match Hallo config |
| Identity split | 90/10 by identity | Standard in literature |
| FVD backbone | I3D (Kinetics pretrained) | Community standard |
| SyncNet | wav2lip or syncnet_python | Most common implementations |
| E-FID | EMOCA or DECA 3DMM | Provides expression parameters |

---

## 7. Gap Analysis Summary

| Component | Hallo Status | VersaFace Action |
|-----------|--------------|------------------|
| Data preprocessing | ✅ Complete | Reuse utilities |
| Face detection/alignment | ✅ Complete | Reuse ImageProcessor |
| Audio processing | ✅ Complete | Reuse AudioProcessor |
| Output format conventions | ✅ Defined | Adopt structure |
| Identity splits | ❌ Missing | Implement from scratch |
| FID computation | ❌ Missing | Implement with clean-fid |
| FVD computation | ❌ Missing | Implement with pytorch-fvd |
| SyncNet metrics | ❌ Missing | Implement with syncnet |
| E-FID computation | ❌ Missing | Implement with EMOCA/DECA |
| Evaluation CLI | ❌ Missing | Implement from scratch |
| Stress slices | ❌ N/A | VersaFace-specific addition |

---

## 8. Next Steps

1. Create canonical manifest schema extending Hallo's metadata format
2. Implement identity-based split generation for HDTF/CelebV
3. Implement metric calculators in `versaface_eval/metrics/`
4. Create generation CLI that produces standardized outputs
5. Create evaluation CLI that computes all metrics
6. Add stress slice generators (silent, non-English, async)
