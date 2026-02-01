"""Quantization format definitions and configurations."""

from enum import Enum
from typing import Dict, Any


class QuantizationFormat(Enum):
    """Supported quantization formats."""
    NVFP4 = "nvfp4"
    FP8 = "fp8"
    INT4_AWQ = "int4_awq"


def get_quantization_config(
    format: QuantizationFormat,
    group_size: int = 16,
    mlp_only: bool = True
) -> Dict[str, Any]:
    """
    Generate quantization config for the specified format.
    
    Args:
        format: Quantization format (NVFP4, FP8, etc.)
        group_size: Block size for group quantization
        mlp_only: Whether to quantize only MLP weights
        
    Returns:
        Dictionary with quantization config for HF/vLLM/TRT-LLM
    """
    exclude_modules = [
        "lm_head",
        "*embed_tokens*",
        "*layernorm*",
        "*norm*",
    ]
    
    if mlp_only:
        exclude_modules.extend([
            "*self_attn*",
            "*q_proj*",
            "*k_proj*",
            "*v_proj*",
            "*o_proj*",
            "*kv_a_proj*",
            "*kv_b_proj*",
            "*q_a_proj*",
            "*q_b_proj*",
        ])
    
    if format == QuantizationFormat.NVFP4:
        return {
            "producer": {
                "name": "modelopt-streaming",
                "version": "0.1.0"
            },
            "quantization": {
                "quant_algo": "NVFP4",
                "kv_cache_quant_algo": None,
                "group_size": group_size,
                "exclude_modules": exclude_modules
            },
            # Compressed-tensors format (for vLLM)
            "config_groups": {
                "group_0": {
                    "input_activations": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": group_size
                    },
                    "weights": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": group_size
                    },
                    "targets": ["Linear"]
                }
            },
            "ignore": exclude_modules,
            "quant_algo": "NVFP4",
            "quant_method": "modelopt"
        }
    
    elif format == QuantizationFormat.FP8:
        raise NotImplementedError("FP8 support coming soon")
    
    elif format == QuantizationFormat.INT4_AWQ:
        raise NotImplementedError("INT4-AWQ support coming soon")
    
    else:
        raise ValueError(f"Unknown format: {format}")
