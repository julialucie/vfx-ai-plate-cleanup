#!/usr/bin/env python3
"""
ControlNet Inpainting Pipeline.
Feeds a masked plate into a ControlNet-constrained Stable Diffusion XL (or Flux)
pipeline so the generative fill is anchored to the structural gradients of the
surrounding, untouched background region.

Composition math:
    I_final = (1 - M) ⊙ I_original + M ⊙ I_generated

The surrounding pixels are NEVER touched — only masked pixels are replaced.
"""

import numpy as np
import torch
from PIL import Image

try:
    from diffusers import (
        StableDiffusionXLControlNetInpaintPipeline,
        ControlNetModel,
        AutoencoderKL,
    )
    from diffusers.utils import load_image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("[WARN] diffusers not installed. ControlNet pipeline unavailable.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_to_srgb_uint8(linear: np.ndarray) -> np.ndarray:
    """Reinhard tonemap + gamma encode for safe uint8 conversion of HDR floats."""
    tonemapped = linear / (linear + 1.0)
    gamma = np.power(np.clip(tonemapped, 0.0, 1.0), 1.0 / 2.2)
    return (gamma * 255.0).astype(np.uint8)


def _build_canny_control_image(rgb_uint8: np.ndarray) -> Image.Image:
    """Extract Canny edges from the plate to guide ControlNet structural alignment."""
    try:
        import cv2
        edges = cv2.Canny(cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY), threshold1=50, threshold2=150)
        edges_rgb = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges_rgb)
    except ImportError:
        # Fallback: return blank control image (ControlNet will rely on inpaint context only)
        print("[WARN] opencv-python not installed. Using blank ControlNet guidance image.")
        return Image.fromarray(np.zeros_like(rgb_uint8))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ControlNetInpaintPipeline:
    """
    Wraps a ControlNet + SDXL inpainting pipeline with:
      - Seed locking for reproducible fills
      - Strict prompt injection for lighting/grain context
      - Edge-aware structural guidance from Canny ControlNet
    """

    def __init__(
        self,
        controlnet_id: str = "diffusers/controlnet-canny-sdxl-1.0",
        base_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        vae_id: str = "madebyoliver/sdxl-vae-fp16-fix",
        device: str | None = None,
        seed: int = 42,
    ):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Install diffusers: pip install diffusers>=0.27.0")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"[INFO] Loading ControlNet weights: {controlnet_id}")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id, torch_dtype=torch.float16
        )

        vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)

        print(f"[INFO] Loading SDXL inpaint backbone: {base_model_id}")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.pipe.enable_model_cpu_offload()
        print("[INFO] ControlNet inpaint pipeline ready.")

    def fill(
        self,
        linear_plate: np.ndarray,
        mask: np.ndarray,
        prompt: str = "film plate background, seamless, matching grain, photorealistic",
        negative_prompt: str = "blurry, watermark, text, artifacts, color shift",
        num_inference_steps: int = 30,
        controlnet_conditioning_scale: float = 0.8,
        strength: float = 0.99,
    ) -> np.ndarray:
        """
        Run ControlNet-guided inpainting and composite the result.

        Args:
            linear_plate: HDR float array  I ∈ R^{H×W×C}  (RGB or RGBA).
            mask:         Binary mask      M ∈ {0,1}^{H×W}  (1 = fill this region).
            prompt:       Positive prompt injected into the diffusion latent space.
            negative_prompt: Tokens the model should steer away from.
            num_inference_steps: Denoising steps (more = higher quality, slower).
            controlnet_conditioning_scale: How strongly edges constrain the fill.
            strength: How much of the masked region to repaint (1.0 = fully).

        Returns:
            Composited linear float array of the same shape as linear_plate,
            with I_final = (1 - M) ⊙ I_original + M ⊙ I_generated.
        """
        rgb_uint8 = _linear_to_srgb_uint8(linear_plate[:, :, :3])
        pil_image = Image.fromarray(rgb_uint8)

        mask_uint8 = (mask * 255).astype(np.uint8)
        pil_mask = Image.fromarray(mask_uint8)

        control_image = _build_canny_control_image(rgb_uint8)

        # Reset generator so the same seed always produces the same fill
        self.generator.manual_seed(self.seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            control_image=control_image,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=strength,
            generator=self.generator,
        ).images[0]

        generated_uint8 = np.array(result)

        # Back to linear float (inverse sRGB gamma, approximate)
        generated_linear = np.power(generated_uint8.astype(np.float32) / 255.0, 2.2)

        # Composite: only masked pixels receive generated content
        m = mask[:, :, np.newaxis].astype(np.float32)
        original_rgb = linear_plate[:, :, :3].astype(np.float32)
        composited_rgb = (1.0 - m) * original_rgb + m * generated_linear

        final = linear_plate.copy().astype(np.float32)
        final[:, :, :3] = composited_rgb

        # Punch alpha transparency in the filled region so comp artists can verify
        if linear_plate.shape[2] == 4:
            final[:, :, 3] = np.where(mask, 0.0, linear_plate[:, :, 3])

        return final


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Run ControlNet inpainting on an EXR plate.")
    parser.add_argument("--input-npy", required=True, help="NumPy .npy file of linear HDR plate (H×W×4).")
    parser.add_argument("--mask-npy", required=True, help="NumPy .npy file of binary mask (H×W).")
    parser.add_argument("--output-npy", required=True, help="Destination .npy for composited plate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    plate = np.load(args.input_npy)
    mask = np.load(args.mask_npy)

    pipe = ControlNetInpaintPipeline(seed=args.seed)
    result = pipe.fill(plate, mask, num_inference_steps=args.steps)

    np.save(args.output_npy, result)
    print(f"[SUCCESS] Composited plate saved to {args.output_npy}")
