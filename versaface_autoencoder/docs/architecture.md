# VersaFace Autoencoder Architecture

This document describes the autoencoder architecture used in VersaFace, based on the Hallo foundation.

## Architecture Overview

VersaFace uses a **latent diffusion** architecture with the following key components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VersaFace Generation Pipeline                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐                                                        │
│  │ Reference    │──→ InsightFace ──→ 512-dim ──→ ImageProjModel ──┐     │
│  │ Image        │    Face Analysis   embedding   (4 context tokens)  │     │
│  └──────────────┘                                                    │     │
│                                                                      │     │
│  ┌──────────────┐                                                    │     │
│  │ Audio        │──→ wav2vec2 ──→ Features ──→ AudioProjModel ──────┤     │
│  │ (Speech)     │    encoder     (12×768)     (32 context tokens)    │     │
│  └──────────────┘                                                    │     │
│                                                                      ▼     │
│  ┌──────────────┐    ┌─────────────────────────────────────────────────┐  │
│  │ Reference    │──→ │                  UNet3D                          │  │
│  │ Image        │    │  (Encoder-Decoder with Motion Modules)          │  │
│  │ (VAE Latent) │    │                                                   │  │
│  └──────────────┘    │  Cross-attention ← Face + Audio context tokens   │  │
│         │            │  Mutual self-attn ← Reference attention maps     │  │
│         │            │  Motion modules ← Temporal consistency           │  │
│         ▼            └─────────────────────────────────────────────────┘  │
│  ┌──────────────┐                            │                            │
│  │ FaceLocator  │──→ Face conditioning ──────┘                            │
│  │ (3D Conv)    │    (320-dim features)                                   │
│  └──────────────┘                                                         │
│                                                                           │
│                                    ▼                                       │
│                           ┌──────────────┐                                │
│                           │ VAE Decoder  │──→ Generated Video Frames     │
│                           │ (Latent→RGB) │                                │
│                           └──────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. VAE (Variational Autoencoder)

**Source:** Diffusers `AutoencoderKL` (sd-vae-ft-mse)

| Property | Value |
|----------|-------|
| Latent channels | 4 |
| Scale factor | 8 (spatial compression) |
| Scaling coefficient | 0.18215 |
| Input resolution | 512×512 |
| Latent resolution | 64×64 |

**Usage:**
```python
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

# Encode image to latent
latent = vae.encode(image).latent_dist.mean * 0.18215

# Decode latent to image
image = vae.decode(latent / 0.18215).sample
```

### 2. FaceLocator (3D Convolutional Encoder)

**File:** `hallo/models/face_locator.py`

Extracts face-conditioned features using 3D convolutions for temporal consistency.

| Layer | Input Channels | Output Channels | Stride |
|-------|----------------|-----------------|--------|
| Conv_in | 3 | 16 | 1 |
| Block 1 | 16 | 32 | 2 |
| Block 2 | 32 | 64 | 2 |
| Block 3 | 64 | 128 | 2 |
| Block 4 | 128 | 320 | 2 |

**Output:** 320-channel feature map at 1/16 spatial resolution

### 3. ImageProjModel (Face Embedding Projection)

**File:** `hallo/models/image_proj.py`

Projects InsightFace embeddings to UNet cross-attention space.

```
Input: face_embeds (batch, 512)
    ↓
Linear: 512 → 4096
    ↓
Reshape: (batch, 4, 1024)
    ↓
LayerNorm
    ↓
Output: clip_extra_context_tokens (batch, 4, 1024)
```

**Purpose:** Creates 4 context tokens for cross-attention identity guidance.

### 4. AudioProjModel (Audio Feature Projection)

**File:** `hallo/models/audio_proj.py`

Projects wav2vec2 audio features to context tokens.

```
Input: audio_embeds (batch, video_length, 5, 12, 768)
    ↓
Flatten: (batch, 5×12×768 = 46080)
    ↓
Linear + ReLU: 46080 → 512
    ↓
Linear + ReLU: 512 → 512
    ↓
Linear: 512 → 24576
    ↓
Reshape: (batch, video_length, 32, 768)
    ↓
LayerNorm
    ↓
Output: context_tokens (batch, video_length, 32, 768)
```

**Purpose:** Creates 32 context tokens per frame for audio-driven lip sync.

### 5. UNet3D (Main Encoder-Decoder)

**File:** `hallo/models/unet_3d.py`

3D UNet with motion modules for video generation.

**Architecture:**
```
ENCODER                         DECODER
---------                       --------
Conv_in (4→320)
    ↓
CrossAttnDown (320) ─────────→ CrossAttnUp (320)
    ↓                                ↑
CrossAttnDown (640) ─────────→ CrossAttnUp (640)
    ↓                                ↑
CrossAttnDown (1280) ────────→ CrossAttnUp (1280)
    ↓                                ↑
DownBlock (1280)                 UpBlock (1280)
    ↓                                ↑
         UNetMidBlock3DCrossAttn
```

**Key Features:**
- **Cross-attention dimension:** 1024
- **Attention heads:** 8
- **Motion modules:** At resolutions [1, 2, 4, 8]
- **Audio integration:** In all attention blocks

### 6. ReferenceAttentionControl (Mutual Self-Attention)

**File:** `hallo/models/mutual_self_attention.py`

Transfers appearance from reference image to generated frames.

**Mechanism:**
1. **Write mode:** Process reference image through UNet, store attention maps
2. **Read mode:** Process generated frames, fuse with stored reference attention

**Applied to:** All blocks (down, mid, up) for full appearance control

## VersaFace Extensions

VersaFace extends the Hallo architecture with:

### 1. Enhanced Expression Control
- Additional expression tokens from EMOCA/DECA 3DMM parameters
- Expression-aware attention modules

### 2. Multi-Scale Face Conditioning
- Face masks at multiple resolutions (64, 32, 16, 8)
- Separate conditioning for: face region, lip region, pose/background

### 3. Improved Temporal Consistency
- Extended motion modules with longer temporal context
- Frame interpolation for smooth transitions

## Training Stages

### Stage 1: Static Image Training
- Train face encoder and reference attention
- No audio conditioning
- Focus on identity preservation

### Stage 2: Audio-Driven Animation
- Add audio projection and lip sync
- Train motion modules
- Focus on temporal coherence

## Configuration

Default configuration in `configs/autoencoder_config.yaml`:

```yaml
vae:
  pretrained: "stabilityai/sd-vae-ft-mse"
  scale_factor: 0.18215

face_encoder:
  embedding_dim: 512  # InsightFace output
  proj_tokens: 4      # Context tokens
  cross_attn_dim: 1024

audio_encoder:
  model: "wav2vec2-base-960h"
  feature_dim: 768
  num_layers: 12
  proj_tokens: 32

unet:
  in_channels: 4
  out_channels: 4
  cross_attention_dim: 1024
  attention_head_dim: 8
  motion_module_resolutions: [1, 2, 4, 8]
```

## References

- Hallo Paper: arXiv 2406.08801
- Stable Diffusion VAE: stabilityai/sd-vae-ft-mse
- wav2vec2: facebook/wav2vec2-base-960h
- InsightFace: https://github.com/deepinsight/insightface
