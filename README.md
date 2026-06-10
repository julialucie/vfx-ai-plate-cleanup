# VFX TensorRT AI Inpaint Pipeline

An industrial-grade, deterministic plate-cleanup and conform pipeline that removes on-set tracking markers, boom mics, and camera rigs from live-action VFX plates — without hallucinating pixels outside the masked region.

---

## The Problem

Artists waste thousands of hours manually painting out tracking markers and on-set rigging. Existing automated AI tools "hallucinate" new textures, shifting background geometry or lighting completely and breaking frame-to-frame continuity. This pipeline solves that.

**The Engineering Problem (low-level):** Professional film pipelines use 32-bit floating-point linear OpenEXR files. Standard neural network encoders expect bounded 8-bit integers — passing raw linear floats directly causes severe numerical clipping errors and model crashes.

---

## System Architecture

```
              [ Raw Live-Action Plate (EXR Sequence) ]
                                 │
                                 ▼
           ┌───────────────────────────────────────────┐
           │    STAGE-1: INGESTION & META CONFORM      │
           │  - Read metadata (Timecode, Camera Lens)  │
           │  - Convert EXR to linear NumPy arrays     │
           └───────────────────────────────────────────┘
                                 │
                                 ▼
           ┌───────────────────────────────────────────┐
           │       STAGE-2: OBJECT MASK ENGINE         │
           │  - Segment Anything Model 2 (SAM 2)       │
           │  - Generate precise black/white mask      │
           └───────────────────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
   [ Mask Channel (B&W) ]         [ Structural Context (Canny) ]
                 │                               │
                 ▼                               ▼
   ┌─────────────────────────┐   ┌─────────────────────────┐
   │  ControlNet Inpainting  │   │    TensorRT Engine      │
   │  (Forces Edge Matching) │   │    (Quantized FP16)     │
   └─────────────────────────┘   └─────────────────────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
           ┌───────────────────────────────────────────┐
           │      STAGE-3: GENERATIVE FILL CORE        │
           │  - Stable Diffusion XL / Flux             │
           │  - Seed locking + strict prompt injection │
           └───────────────────────────────────────────┘
                                 │
                                 ▼
           ┌───────────────────────────────────────────┐
           │         STAGE-4: POST-PROCESSING          │
           │  - Composite blending (edge feather)      │
           │  - Re-inject original high-frequency grain│
           └───────────────────────────────────────────┘
                                 │
                                 ▼
             [ Clean Finished Plate (Conformed EXR) ]
```

---

## Detailed Implementation

### 1. Ingestion & Pre-Processing (`conformed_pipeline.py`)

Reads uncompressed `.exr` metadata headers to resolve dynamic image resolutions. The ingestion loop converts the file's native planar layer structure into an interleaved multi-channel floating-point data tensor:

$$\mathbf{I} \in \mathbb{R}^{H \times W \times C}$$

where C includes red, green, blue, and alpha channels stored as raw 32-bit floats.

### 2. Masking & Isolation (`conformed_pipeline.py`)

A localized Reinhard tonemapping curve scales infinite HDR float values safely down to bounded uint8 format before passing to SAM. The model outputs a binary mask array:

$$\mathbf{M} \in \{0, 1\}^{H \times W}$$

where `1` marks the artifact to remove.

### 3. Low-Level Optimization (`tensorrt_engine.py`)

Stock Hugging Face diffusion models run at ~4.5 seconds per frame on an A100. The TensorRT engine cuts this to **under 400 ms**:

- Export model weights to ONNX
- Build a serialized TensorRT engine with FP16 quantization
- Pre-allocate static pinned host + device GPU buffers — eliminates per-frame `malloc` overhead

```bash
python tensorrt_engine.py --onnx model.onnx --engine model_fp16.trt
```

| Mode | Latency (mean) | VRAM |
|---|---|---|
| HuggingFace FP32 (baseline) | 4500 ms | ~18 GB |
| PyTorch FP16 | ~1200 ms | ~10 GB |
| **TensorRT FP16 (this pipeline)** | **<400 ms** | **~6 GB** |

### 4. Generative Context Integration (`controlnet_pipeline.py`)

The masked plate passes into a ControlNet pipeline. ControlNet reads the structural gradients (Canny edges) of the unmasked surrounding area and forces the generative fill to align with the existing lighting, grain, and lens distortion.

The final composite is a strict masking matrix operation — **surrounding pixels are mathematically guaranteed to be untouched**:

$$\mathbf{I}_{\text{final}} = (1 - \mathbf{M}) \odot \mathbf{I}_{\text{original}} + \mathbf{M} \odot \mathbf{I}_{\text{generated}}$$

### 5. Studio Automation Output (`api_server.py` + `nuke_gizmo/`)

A FastAPI headless microservice exposes a single `POST /cleanup` endpoint. A custom Foundry Nuke Python gizmo sends the EXR path via JSON and receives the conformed plate path directly into the artist's local cache.

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/cleanup \
  -H "Content-Type: application/json" \
  -d '{"exr_path": "/mnt/shots/shot_010.exr", "coord_x": 500, "coord_y": 300}'
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the CLI (single frame)

```bash
python conformed_pipeline.py \
  --input shot_010.exr \
  --output conformed_shot_010.exr \
  --x 500 --y 300
```

### 3. Run the ControlNet inpaint pipeline

```bash
python controlnet_pipeline.py \
  --input-npy plate.npy \
  --mask-npy mask.npy \
  --output-npy result.npy \
  --seed 42 --steps 30
```

### 4. Start the API server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### 5. Build a TensorRT engine

```bash
python tensorrt_engine.py --onnx sdxl_inpaint.onnx --engine sdxl_inpaint_fp16.trt
```

### 6. Nuke gizmo

Copy `nuke_gizmo/plate_cleanup_button.py` to your `~/.nuke/` directory and add to `menu.py`:

```python
import plate_cleanup_button
toolbar = nuke.menu("Nodes")
m = toolbar.addMenu("VFX AI")
m.addCommand("Plate Cleanup", "plate_cleanup_button.run()")
```

---

## Docker

```dockerfile
FROM nvidia/cuda:12.3.0-cudnn9-runtime-ubuntu22.04
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t plate-cleanup-api .
docker run --gpus all -p 8000:8000 \
  -e SAM_CHECKPOINT=/checkpoints/sam_vit_h_4b8939.pth \
  -v /mnt/checkpoints:/checkpoints \
  plate-cleanup-api
```

---

## Repository Structure

```
vfx-ai-plate-cleanup/
├── conformed_pipeline.py      # Stage 1–2: EXR ingestion + SAM masking
├── controlnet_pipeline.py     # Stage 3: ControlNet + SDXL inpainting
├── tensorrt_engine.py         # TRT engine builder + FP16 inference runtime
├── api_server.py              # FastAPI microservice backend
├── nuke_gizmo/
│   └── plate_cleanup_button.py  # Foundry Nuke Python gizmo
└── requirements.txt
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SAM_CHECKPOINT` | `sam_vit_h_4b8939.pth` | Path to SAM model weights |
| `SAM_MODEL_TYPE` | `vit_h` | SAM model variant |
| `INPAINT_SEED` | `42` | Diffusion seed for reproducible fills |
| `INPAINT_STEPS` | `30` | Denoising steps (quality vs speed) |
| `PLATE_CACHE_DIR` | `/tmp/plate_cleanup_cache` | Output directory for conformed EXRs |
| `PLATE_CLEANUP_API_URL` | `http://localhost:8000/cleanup` | API URL used by the Nuke gizmo |
