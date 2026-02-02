"""NVFP4 quantization implementation using Model-Optimizer primitives."""

from typing import Optional

import torch


def quantize_weight_nvfp4(
    weight: torch.Tensor,
    NVFP4QTensor,
    block_size: int = 16,
    device: str = "cuda:0",
    shared_scale_2: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a single weight tensor to NVFP4 format.
    
    Uses NVIDIA Model-Optimizer's NVFP4QTensor.quantize() to perform the actual
    quantization, ensuring compatibility with TensorRT-LLM and vLLM.
    
    Args:
        weight: Input weight tensor (BF16/FP16)
        NVFP4QTensor: Model-Optimizer's NVFP4QTensor class
        block_size: Block size for group quantization (default: 16)
        device: CUDA device for quantization kernels
        shared_scale_2: Optional pre-computed weight_scale_2 (for w1/w3 vLLM compatibility)
        
    Returns:
        Tuple of:
            - packed_weight (uint8): Packed NVFP4 weights (2 FP4 values per byte)
            - weight_scale (float8_e4m3fn): Per-block scaling factors
            - weight_scale_2 (float32): Global (per-tensor) scaling factor
    """
    weight = weight.to(device)
    
    # Call NVFP4QTensor.quantize directly
    # This computes:
    #   - weight_scale_2 = global_max(|weight|) / (6.0 * 448.0)  [or use shared_scale_2]
    #   - weight_scale = per_block_max(|weight|) / (6.0 * weight_scale_2)
    #   - quantized = pack_to_fp4(weight / (weight_scale * weight_scale_2))
    quantized, weight_scale, weight_scale_2 = NVFP4QTensor.quantize(
        weight,
        block_size=block_size,
        weights_scaling_factor=None,  # Computed automatically
        weights_scaling_factor_2=shared_scale_2,  # Use shared if provided
        keep_high_precision=False,
    )
    
    # Extract packed data and move to CPU
    packed_weight = quantized._quantized_data.to("cpu", non_blocking=True)
    weight_scale = weight_scale.to("cpu", non_blocking=True)
    weight_scale_2 = weight_scale_2.to("cpu", non_blocking=True)
    
    return packed_weight, weight_scale, weight_scale_2


def compute_dummy_input_scale(weight_shape: tuple) -> torch.Tensor:
    """
    Compute a dummy input_scale for weight-only quantization.
    
    In weight-only mode, we don't calibrate activations, so we use a neutral
    scaling factor (1.0) that acts as a no-op for runtime activation quantization.
    
    Args:
        weight_shape: Shape of the weight tensor
        
    Returns:
        Tensor containing the dummy input scale (1.0)
    """
    return torch.tensor(1.0, dtype=torch.float32)
