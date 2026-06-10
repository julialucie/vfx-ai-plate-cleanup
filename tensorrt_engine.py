#!/usr/bin/env python3
"""
TensorRT FP16 Inference Engine.
Converts Hugging Face diffusion model weights to an optimized TensorRT engine
with static memory allocation for minimal VRAM overhead.

Target: reduce per-frame inference from ~4.5s (stock HuggingFace) to <400ms.
"""

import os
import time
import numpy as np
import torch

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 — initializes CUDA context on import
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("[WARN] TensorRT not found. Falling back to PyTorch FP16 inference.")


TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if TRT_AVAILABLE else None


# ---------------------------------------------------------------------------
# Engine Build
# ---------------------------------------------------------------------------

def build_engine_from_onnx(onnx_path: str, engine_path: str, use_fp16: bool = True) -> None:
    """
    Convert an ONNX model to a serialized TensorRT engine file.

    Args:
        onnx_path:   Path to the exported ONNX model.
        engine_path: Destination path for the .trt engine file.
        use_fp16:    Enable FP16 quantization (halves VRAM footprint).
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is required to build an engine.")

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB workspace

        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 quantization enabled.")

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[ERROR] ONNX parse error: {parser.get_error(i)}")
                raise RuntimeError("ONNX parsing failed.")

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed.")

        with open(engine_path, "wb") as f:
            f.write(serialized)

    print(f"[INFO] Engine saved to {engine_path}")


# ---------------------------------------------------------------------------
# Engine Runtime
# ---------------------------------------------------------------------------

class TRTInferenceEngine:
    """
    Loads a serialized TensorRT engine and runs inference with pre-allocated
    static GPU buffers — eliminates per-frame malloc overhead.
    """

    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is required to run TRTInferenceEngine.")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self._allocate_buffers()
        print(f"[INFO] TensorRT engine loaded: {engine_path}")

    def _allocate_buffers(self):
        """Pre-allocate pinned host + device buffers for all I/O tensors."""
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        """
        Run a single forward pass.

        Args:
            input_array: Pre-processed numpy array matching the engine's input shape.

        Returns:
            Output numpy array from the engine's first output binding.
        """
        np.copyto(self.inputs[0]["host"], input_array.ravel())

        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()

        return self.outputs[0]["host"].copy()

    def benchmark(self, input_array: np.ndarray, iterations: int = 50) -> dict:
        """Warm-up then time N inference passes. Returns latency stats in ms."""
        for _ in range(5):
            self.infer(input_array)

        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.infer(input_array)
            times.append((time.perf_counter() - t0) * 1000)

        return {
            "mean_ms": float(np.mean(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p95_ms": float(np.percentile(times, 95)),
        }


# ---------------------------------------------------------------------------
# PyTorch FP16 fallback (no TensorRT dependency)
# ---------------------------------------------------------------------------

class TorchFP16Engine:
    """
    Thin wrapper around a PyTorch model running in FP16 on CUDA.
    Used when TensorRT is unavailable (e.g., CPU-only dev machines).
    """

    def __init__(self, model: torch.nn.Module):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.half().to(self.device).eval()
        print(f"[INFO] PyTorch FP16 fallback engine on {self.device.upper()}.")

    @torch.inference_mode()
    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor.half().to(self.device))


# ---------------------------------------------------------------------------
# CLI utility: export ONNX then build TRT engine
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX model.")
    parser.add_argument("--onnx", required=True, help="Path to input ONNX model.")
    parser.add_argument("--engine", required=True, help="Destination path for .trt engine file.")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 quantization (use FP32).")
    args = parser.parse_args()

    build_engine_from_onnx(args.onnx, args.engine, use_fp16=not args.no_fp16)
