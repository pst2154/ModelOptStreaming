#!/usr/bin/env python3
"""Compare scale distributions between NVIDIA and streaming NVFP4 models."""

import torch
from safetensors import safe_open
from pathlib import Path
import numpy as np

def analyze_scales(model_path: str, model_name: str):
    """Analyze scale distributions in an NVFP4 model."""
    model_path = Path(model_path)
    
    # Find first shard
    shards = sorted(model_path.glob("model-*.safetensors"))
    if not shards:
        print(f"No shards found in {model_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Path: {model_path}")
    print(f"Total shards: {len(shards)}")
    print(f"{'='*60}")
    
    # Collect scales from first 3 shards
    weight_scale_2_values = []
    weight_scale_values = []
    input_scale_values = []
    gate_up_scale2_pairs = []
    
    for shard_idx, shard_file in enumerate(shards[:3]):
        print(f"\nAnalyzing {shard_file.name}...")
        
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            for key in keys:
                if key.endswith("_scale_2"):
                    scale_2 = f.get_tensor(key).item()  # Should be scalar
                    weight_scale_2_values.append(scale_2)
                    
                    # Check for gate/up pairs
                    if "gate_proj.weight_scale_2" in key:
                        layer_prefix = key.replace(".gate_proj.weight_scale_2", "")
                        up_key = f"{layer_prefix}.up_proj.weight_scale_2"
                        if up_key in keys:
                            up_scale_2 = f.get_tensor(up_key).item()
                            matches = abs(scale_2 - up_scale_2) < 1e-9
                            gate_up_scale2_pairs.append((key, scale_2, up_scale_2, matches))
                
                elif key.endswith("_scale") and not key.endswith("_scale_2"):
                    scale = f.get_tensor(key)
                    # This is per-block scale (float8_e4m3fn)
                    weight_scale_values.extend(scale.flatten().float().tolist()[:100])  # Sample 100
                
                elif key.endswith(".input_scale"):
                    input_scale = f.get_tensor(key).item()
                    input_scale_values.append(input_scale)
    
    print(f"\n{'='*60}")
    print(f"SCALE STATISTICS")
    print(f"{'='*60}")
    
    if weight_scale_2_values:
        arr = np.array(weight_scale_2_values)
        print(f"\nweight_scale_2 (global, float32 scalar):")
        print(f"  Count: {len(arr)}")
        print(f"  Min:   {arr.min():.6e}")
        print(f"  Max:   {arr.max():.6e}")
        print(f"  Mean:  {arr.mean():.6e}")
        print(f"  Std:   {arr.std():.6e}")
    
    if weight_scale_values:
        arr = np.array(weight_scale_values)
        print(f"\nweight_scale (per-block, float8_e4m3fn) [sample of 100]:")
        print(f"  Count: {len(arr)}")
        print(f"  Min:   {arr.min():.6e}")
        print(f"  Max:   {arr.max():.6e}")
        print(f"  Mean:  {arr.mean():.6e}")
        print(f"  Std:   {arr.std():.6e}")
    
    if input_scale_values:
        arr = np.array(input_scale_values)
        print(f"\ninput_scale (activation, float32 scalar):")
        print(f"  Count: {len(arr)}")
        print(f"  All equal to 1.0: {np.allclose(arr, 1.0)}")
        print(f"  Min:   {arr.min():.6e}")
        print(f"  Max:   {arr.max():.6e}")
    
    if gate_up_scale2_pairs:
        print(f"\n{'='*60}")
        print(f"GATE/UP SCALE_2 MATCHING (critical for vLLM)")
        print(f"{'='*60}")
        print(f"Total gate/up pairs found: {len(gate_up_scale2_pairs)}")
        
        matches = sum(1 for _, _, _, m in gate_up_scale2_pairs if m)
        print(f"Matched pairs: {matches}/{len(gate_up_scale2_pairs)}")
        
        if matches < len(gate_up_scale2_pairs):
            print(f"\n⚠️  WARNING: Found {len(gate_up_scale2_pairs) - matches} mismatched gate/up pairs!")
            for gate_key, gate_s2, up_s2, matched in gate_up_scale2_pairs[:5]:
                if not matched:
                    print(f"  {gate_key}:")
                    print(f"    gate: {gate_s2:.6e}")
                    print(f"    up:   {up_s2:.6e}")
                    print(f"    diff: {abs(gate_s2 - up_s2):.6e}")
        else:
            print(f"✅ All gate/up pairs have matching weight_scale_2!")
            # Show a few examples
            for gate_key, gate_s2, up_s2, _ in gate_up_scale2_pairs[:3]:
                print(f"  {gate_key}: {gate_s2:.6e}")

if __name__ == "__main__":
    import sys
    
    nvidia_path = "/lustre/fsw/portfolios/general/users/asteiner/Kimi-K2.5-NVFP4-NVIDIA"
    streaming_path = "/lustre/fsw/portfolios/general/users/asteiner/Kimi-K2.5-NVFP4-NO-CALIBRATION-20260202_111729"
    
    analyze_scales(nvidia_path, "NVIDIA Official")
    analyze_scales(streaming_path, "Streaming (Ours)")
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")
