#!/usr/bin/env python3
"""
Industrial-Grade AI Plate Cleanup Core Pipeline.
Bridges OpenEXR Ingestion, Neural Target Location via SAM, and Matrix Composition.
"""

import os
import argparse
import numpy as np
import OpenEXR
import Imath
import torch
from segment_anything import sam_model_registry, SamPredictor

class AIPlateCleanupPipeline:
    def __init__(self, checkpoint_path="sam_vit_h_4b8939.pth", model_type="vit_h"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Initializing Core Neural Engine on hardware device: {self.device.upper()}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing model weights at {checkpoint_path}. Please download them first.")
            
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(self.device)
        self.predictor = SamPredictor(self.sam)

    def read_linear_hdr_plate(self, exr_path):
        """MODULE 1: Stream-reads linear EXR floats into raw multi-channel arrays."""
        file = OpenEXR.InputFile(exr_path)
        dw = file.header()['dataWindow']
        shape = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

        channels = ['R', 'G', 'B']
        frames = [np.frombuffer(file.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(shape) for c in channels]
        
        if 'A' in file.header()['channels']:
            alpha = np.frombuffer(file.channel('A', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(shape)
            channels_to_stack = frames + [alpha]
        else:
            channels_to_stack = frames + [np.ones_like(frames)]

        file.close()
        return np.stack(channels_to_stack, axis=-1), shape

    def generate_binary_stencil(self, linear_plate, coord_x, coord_y):
        """MODULE 2: Safely maps HDR values to uint8 space and generates the binary mask."""
        rgb_linear = np.clip(linear_plate[:, :, :3], 0.0, None)
        rgb_uint8 = ((rgb_linear / (rgb_linear + 1.0)) * 255.0).astype(np.uint8)
        
        self.predictor.set_image(rgb_uint8)
        masks, _, _ = self.predictor.predict(
            point_coords=np.array([[coord_x, coord_y]]), 
            point_labels=np.array(), 
            multimask_output=False
        )
        return masks

    def write_conformed_asset(self, output_path, img_array, spatial_shape):
        """MODULE 3: Pack and save modified float arrays back out to a production-valid EXR."""
        header = OpenEXR.Header(spatial_shape, spatial_shape)
        half_type = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = {'R': half_type, 'G': half_type, 'B': half_type, 'A': half_type}

        pixel_data = {
            'R': img_array[:, :, 0].astype(np.float16).tobytes(),
            'G': img_array[:, :, 1].astype(np.float16).tobytes(),
            'B': img_array[:, :, 2].astype(np.float16).tobytes(),
            'A': img_array[:, :, 3].astype(np.float16).tobytes()
        }
        
        exr_out = OpenEXR.OutputFile(output_path, header)
        exr_out.writePixels(pixel_data)
        exr_out.close()

def main():
    parser = argparse.ArgumentParser(description="Core AI Plate Cleanup Utility.")
    parser.add_argument("--input", type=str, required=True, help="Path to input EXR plate.")
    parser.add_argument("--output", type=str, required=True, help="Path to save conformed EXR asset.")
    parser.add_argument("--x", type=int, required=True, help="Tracking coordinate X point.")
    parser.add_argument("--y", type=int, required=True, help="Tracking coordinate Y point.")
    args = parser.parse_args()

    pipeline = AIPlateCleanupPipeline()
    
    # Run data loop
    hdr_data, spatial_shape = pipeline.read_linear_hdr_plate(args.input)
    mask = pipeline.generate_binary_stencil(hdr_data, args.x, args.y)
    mask_bool = mask.astype(bool)
    
    # Execute Matrix Composition Blending Logic
    conformed_plate = hdr_data.copy()
    conformed_plate[mask_bool, :3] = 0.0  
    conformed_plate[mask_bool, 3] = 0.0   # Punch transparency opening in the alpha layer
    
    pipeline.write_conformed_asset(args.output, conformed_plate, spatial_shape)
    print(f"[SUCCESS] Conformed production asset saved cleanly to: {args.output}")

if __name__ == "__main__":
    main()
