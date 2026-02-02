#!/usr/bin/env python3
"""Compare actual quantized weight values between NVIDIA and streaming NVFP4 models."""

import torch
from safetensors import safe_open
from pathlib import Path
import sys

def compare_tensor(nvidia_path: Path, streaming_path: Path, weight_key: str):
    """Compare a specific weight tensor between two models."""
    
    print(f"\n{'='*60}")
    print(f"Comparing: {weight_key}")
    print(f"{'='*60}")
    
    # Load from NVIDIA model
    nvidia_shards = sorted(nvidia_path.glob("model-*.safetensors"))
    nvidia_tensor = None
    nvidia_scale = None
    nvidia_scale_2 = None
    nvidia_input_scale = None
    
    for shard in nvidia_shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            if weight_key in f.keys():
                nvidia_tensor = f.get_tensor(weight_key)
                
                # Get associated scales
                scale_key = weight_key.replace(".weight", ".weight_scale")
                scale_2_key = weight_key.replace(".weight", ".weight_scale_2")
                input_scale_key = weight_key.replace(".weight", ".input_scale")
                
                if scale_key in f.keys():
                    nvidia_scale = f.get_tensor(scale_key)
                if scale_2_key in f.keys():
                    nvidia_scale_2 = f.get_tensor(scale_2_key)
                if input_scale_key in f.keys():
                    nvidia_input_scale = f.get_tensor(input_scale_key)
                break
    
    # Load from streaming model
    streaming_shards = sorted(streaming_path.glob("model-*.safetensors"))
    streaming_tensor = None
    streaming_scale = None
    streaming_scale_2 = None
    streaming_input_scale = None
    
    for shard in streaming_shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            if weight_key in f.keys():
                streaming_tensor = f.get_tensor(weight_key)
                
                scale_key = weight_key.replace(".weight", ".weight_scale")
                scale_2_key = weight_key.replace(".weight", ".weight_scale_2")
                input_scale_key = weight_key.replace(".weight", ".input_scale")
                
                if scale_key in f.keys():
                    streaming_scale = f.get_tensor(scale_key)
                if scale_2_key in f.keys():
                    streaming_scale_2 = f.get_tensor(scale_2_key)
                if input_scale_key in f.keys():
                    streaming_input_scale = f.get_tensor(input_scale_key)
                break
    
    if nvidia_tensor is None or streaming_tensor is None:
        print(f"❌ Weight not found in one or both models")
        return
    
    print(f"\n📊 Weight Tensor:")
    print(f"  NVIDIA:    shape={nvidia_tensor.shape}, dtype={nvidia_tensor.dtype}")
    print(f"  Streaming: shape={streaming_tensor.shape}, dtype={streaming_tensor.dtype}")
    print(f"  Shape match: {nvidia_tensor.shape == streaming_tensor.shape}")
    
    # Compare first 20 bytes
    nvidia_bytes = nvidia_tensor.flatten()[:20]
    streaming_bytes = streaming_tensor.flatten()[:20]
    print(f"\n  First 20 packed values:")
    print(f"    NVIDIA:    {nvidia_bytes.tolist()}")
    print(f"    Streaming: {streaming_bytes.tolist()}")
    print(f"    Identical: {torch.equal(nvidia_bytes, streaming_bytes)}")
    
    # Compare scales
    if nvidia_scale_2 is not None and streaming_scale_2 is not None:
        print(f"\n📊 weight_scale_2 (global):")
        nv_s2 = nvidia_scale_2.item() if nvidia_scale_2.numel() == 1 else nvidia_scale_2
        st_s2 = streaming_scale_2.item() if streaming_scale_2.numel() == 1 else streaming_scale_2
        print(f"  NVIDIA:    {nv_s2:.6e}")
        print(f"  Streaming: {st_s2:.6e}")
        if isinstance(nv_s2, float) and isinstance(st_s2, float):
            diff = abs(nv_s2 - st_s2)
            rel_diff = diff / max(abs(nv_s2), 1e-12)
            print(f"  Diff:      {diff:.6e} (rel: {rel_diff:.2%})")
            print(f"  Match:     {diff < 1e-9}")
    
    if nvidia_scale is not None and streaming_scale is not None:
        print(f"\n📊 weight_scale (per-block):")
        print(f"  NVIDIA:    shape={nvidia_scale.shape}, dtype={nvidia_scale.dtype}")
        print(f"  Streaming: shape={streaming_scale.shape}, dtype={streaming_scale.dtype}")
        
        # Compare statistics
        nv_s = nvidia_scale.flatten().float()
        st_s = streaming_scale.flatten().float()
        print(f"  NVIDIA stats:    min={nv_s.min():.6e}, max={nv_s.max():.6e}, mean={nv_s.mean():.6e}")
        print(f"  Streaming stats: min={st_s.min():.6e}, max={st_s.max():.6e}, mean={st_s.mean():.6e}")
        
        # Sample comparison
        print(f"  First 10 blocks (NVIDIA):    {nv_s[:10].tolist()}")
        print(f"  First 10 blocks (Streaming): {st_s[:10].tolist()}")
        print(f"  Exact match: {torch.equal(nvidia_scale, streaming_scale)}")
    
    if nvidia_input_scale is not None and streaming_input_scale is not None:
        print(f"\n📊 input_scale (activation):")
        nv_is = nvidia_input_scale.item()
        st_is = streaming_input_scale.item()
        print(f"  NVIDIA:    {nv_is:.6e}")
        print(f"  Streaming: {st_is:.6e}")
        print(f"  Match:     {abs(nv_is - st_is) < 1e-9}")

if __name__ == "__main__":
    nvidia_path = Path("/lustre/fsw/portfolios/general/users/asteiner/Kimi-K2.5-NVFP4-NVIDIA")
    streaming_path = Path("/lustre/fsw/portfolios/general/users/asteiner/Kimi-K2.5-NVFP4-NO-CALIBRATION-20260202_111729")
    
    # Test a few representative weights (use correct key format)
    test_weights = [
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.10.mlp.gate_proj.weight",
        "language_model.model.layers.10.mlp.up_proj.weight",
    ]
    
    for weight_key in test_weights:
        try:
            compare_tensor(nvidia_path, streaming_path, weight_key)
        except Exception as e:
            print(f"\n❌ Error comparing {weight_key}: {e}")
    
    print(f"\n\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")
