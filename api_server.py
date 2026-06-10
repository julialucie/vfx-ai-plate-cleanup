#!/usr/bin/env python3
"""
FastAPI Headless Microservice — Plate Cleanup Backend.

Exposes a single POST /cleanup endpoint that accepts a JSON payload with
an image path (or base64 blob), runs the full pipeline, and returns the
path to the conformed EXR in the artist's local cache directory.

Designed to be called by:
  - A custom Foundry Nuke Python gizmo (see nuke_gizmo/plate_cleanup_button.py)
  - Any ILM / pipeline asset manager script via standard HTTP
  - Docker / Kubernetes sidecar in a render farm environment

Run locally:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Docker:
    docker build -t plate-cleanup-api .
    docker run --gpus all -p 8000:8000 plate-cleanup-api
"""

import os
import uuid
import base64
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from conformed_pipeline import AIPlateCleanupPipeline
from controlnet_pipeline import ControlNetInpaintPipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.environ.get("PLATE_CACHE_DIR", "/tmp/plate_cleanup_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_h")
INPAINT_SEED = int(os.environ.get("INPAINT_SEED", 42))
INPAINT_STEPS = int(os.environ.get("INPAINT_STEPS", 30))


# ---------------------------------------------------------------------------
# App & lazy-loaded models
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VFX AI Plate Cleanup API",
    description="TensorRT-accelerated inpainting microservice for live-action plate cleanup.",
    version="1.0.0",
)

_sam_pipeline: AIPlateCleanupPipeline | None = None
_controlnet_pipeline: ControlNetInpaintPipeline | None = None


def _get_sam() -> AIPlateCleanupPipeline:
    global _sam_pipeline
    if _sam_pipeline is None:
        _sam_pipeline = AIPlateCleanupPipeline(
            checkpoint_path=SAM_CHECKPOINT,
            model_type=SAM_MODEL_TYPE,
        )
    return _sam_pipeline


def _get_controlnet() -> ControlNetInpaintPipeline:
    global _controlnet_pipeline
    if _controlnet_pipeline is None:
        _controlnet_pipeline = ControlNetInpaintPipeline(seed=INPAINT_SEED)
    return _controlnet_pipeline


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CleanupRequest(BaseModel):
    """
    Payload sent by the Nuke gizmo or pipeline script.
    Either exr_path or exr_base64 must be provided.
    """
    exr_path: str | None = Field(None, description="Absolute path to the input EXR on shared storage.")
    exr_base64: str | None = Field(None, description="Base64-encoded EXR bytes (for non-shared-storage setups).")
    coord_x: int = Field(..., description="X coordinate of the artifact to remove (screen space).")
    coord_y: int = Field(..., description="Y coordinate of the artifact to remove (screen space).")
    prompt: str = Field(
        "film plate background, seamless, matching grain, photorealistic",
        description="Positive diffusion prompt to guide the inpainting fill.",
    )
    output_path: str | None = Field(
        None,
        description="Override output EXR path. Defaults to an auto-named file in CACHE_DIR.",
    )


class CleanupResponse(BaseModel):
    job_id: str
    output_path: str
    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/cleanup", response_model=CleanupResponse)
def cleanup(request: CleanupRequest, background_tasks: BackgroundTasks):
    """
    Main cleanup endpoint. Runs the full SAM → ControlNet → composite pipeline
    and returns the path to the finished, conformed EXR.
    """
    job_id = str(uuid.uuid4())

    # --- Resolve input EXR ---
    if request.exr_path:
        if not os.path.exists(request.exr_path):
            raise HTTPException(status_code=404, detail=f"EXR not found: {request.exr_path}")
        input_exr = request.exr_path
        _tmp_file = None
    elif request.exr_base64:
        _tmp_file = tempfile.NamedTemporaryFile(suffix=".exr", delete=False)
        _tmp_file.write(base64.b64decode(request.exr_base64))
        _tmp_file.flush()
        input_exr = _tmp_file.name
    else:
        raise HTTPException(status_code=422, detail="Provide either exr_path or exr_base64.")

    # --- Resolve output path ---
    output_exr = request.output_path or str(CACHE_DIR / f"{job_id}_conformed.exr")

    try:
        sam = _get_sam()
        hdr_data, spatial_shape = sam.read_linear_hdr_plate(input_exr)

        mask_3d = sam.generate_binary_stencil(hdr_data, request.coord_x, request.coord_y)
        mask_2d = mask_3d[0].astype(np.float32)

        controlnet = _get_controlnet()
        composited = controlnet.fill(hdr_data, mask_2d, prompt=request.prompt, num_inference_steps=INPAINT_STEPS)

        sam.write_conformed_asset(output_exr, composited, spatial_shape)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if _tmp_file is not None:
            os.unlink(_tmp_file.name)

    return CleanupResponse(job_id=job_id, output_path=output_exr, status="complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
