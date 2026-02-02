#!/usr/bin/env python3
"""Validate that ALL gate/up projection pairs have matching weight_scale_2."""

import torch
from safetensors import safe_open
from pathlib import Path
from collections import defaultdict

def validate_gate_up_matching(model_path: str, model_name: str):
    """Check all gate/up pairs for matching weight_scale_2."""
    model_path = Path(model_path)
    shards = sorted(model_path.glob("model-*.safetensors"))
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")
    
    gate_scales = {}  # {layer_prefix: scale_2_value}
    up_scales = {}
    
    # Collect all gate/up scales
    for shard_file in shards:
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "gate_proj.weight_scale_2" in key:
                    layer_prefix = key.replace(".gate_proj.weight_scale_2", "")
                    scale_2 = f.get_tensor(key).item()
                    gate_scales[layer_prefix] = scale_2
                
                elif "up_proj.weight_scale_2" in key:
                    layer_prefix = key.replace(".up_proj.weight_scale_2", "")
                    scale_2 = f.get_tensor(key).item()
                    up_scales[layer_prefix] = scale_2
    
    print(f"\nFound {len(gate_scales)} gate_proj weights")
    print(f"Found {len(up_scales)} up_proj weights")
    
    # Compare
    all_layers = sorted(set(gate_scales.keys()) | set(up_scales.keys()))
    print(f"\nTotal layers to check: {len(all_layers)}")
    
    mismatches = []
    matches = []
    
    for layer_prefix in all_layers:
        gate_s2 = gate_scales.get(layer_prefix)
        up_s2 = up_scales.get(layer_prefix)
        
        if gate_s2 is None or up_s2 is None:
            print(f"⚠️  {layer_prefix}: Missing gate or up scale_2")
            continue
        
        diff = abs(gate_s2 - up_s2)
        rel_diff = diff / max(abs(gate_s2), 1e-12)
        
        if diff < 1e-9:
            matches.append((layer_prefix, gate_s2, up_s2))
        else:
            mismatches.append((layer_prefix, gate_s2, up_s2, diff, rel_diff))
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"✅ Matched pairs: {len(matches)}/{len(all_layers)}")
    print(f"❌ Mismatched pairs: {len(mismatches)}/{len(all_layers)}")
    
    if mismatches:
        print(f"\n⚠️  MISMATCHED PAIRS (showing first 10):")
        for layer_prefix, gate_s2, up_s2, diff, rel_diff in mismatches[:10]:
            print(f"\n  {layer_prefix}:")
            print(f"    gate_proj: {gate_s2:.10e}")
            print(f"    up_proj:   {up_s2:.10e}")
            print(f"    abs diff:  {diff:.10e}")
            print(f"    rel diff:  {rel_diff:.2%}")
    else:
        print(f"\n✅ ALL GATE/UP PAIRS PERFECTLY MATCHED!")
        print(f"   This is required for vLLM's fused MoE kernels.")
        
        # Show a few examples
        print(f"\n   Examples (first 3):")
        for layer_prefix, gate_s2, _ in matches[:3]:
            print(f"     {layer_prefix}: {gate_s2:.10e}")
    
    return len(mismatches) == 0

if __name__ == "__main__":
    nvidia_path = "/lustre/fsw/portfolios/general/users/asteiner/Kimi-K2.5-NVFP4-NVIDIA"
    streaming_path = "/lustre/fsw/portfolios/general/users/asteiner/Kimi-K2.5-NVFP4-NO-CALIBRATION-20260202_111729"
    
    nvidia_ok = validate_gate_up_matching(nvidia_path, "NVIDIA Official")
    streaming_ok = validate_gate_up_matching(streaming_path, "Streaming (Ours)")
    
    print(f"\n\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}")
    print(f"NVIDIA model:    {'✅ PASS' if nvidia_ok else '❌ FAIL'}")
    print(f"Streaming model: {'✅ PASS' if streaming_ok else '❌ FAIL'}")
    print(f"{'='*60}")
