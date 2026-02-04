"""INT4 to BF16 decompression for compressed-tensors format."""

import gc
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def _is_text_only_key(key: str) -> bool:
    """Check if a tensor key belongs to text-only model (not vision)."""
    vision_patterns = [
        "vision_model", "vision_projection", "vision_tower",
        "mm_projector", "visual", "image_encoder", "vision_embed_tokens",
    ]
    key_lower = key.lower()
    return not any(pattern in key_lower for pattern in vision_patterns)


def _existing_shard_count(output_path: str) -> int:
    """Count existing output shards."""
    if not os.path.isdir(output_path):
        return 0
    idxs = []
    for name in os.listdir(output_path):
        m = re.match(r"^model-(\d{5})-of-(?:XXXXX|\d{5})\.safetensors$", name)
        if m:
            idxs.append(int(m.group(1)))
    return max(idxs, default=0)


def _output_shard_name(shard_idx: int, total_shards: int) -> str:
    """Generate output shard filename."""
    return f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"


def _find_existing_shard_path(output_path: str, shard_idx: int) -> Optional[str]:
    """Find existing shard path, supporting both finalized and legacy names."""
    if not os.path.isdir(output_path):
        return None
    # Prefer finalized names
    for p in Path(output_path).glob(f"model-{shard_idx:05d}-of-*.safetensors"):
        return str(p)
    # Check legacy naming
    legacy = os.path.join(output_path, f"model-{shard_idx:05d}-of-XXXXX.safetensors")
    if os.path.exists(legacy):
        return legacy
    return None


def _expected_output_keys_for_file(model_path: str, input_file: str) -> List[str]:
    """Compute output tensor names for a given input file without loading tensors."""
    path = os.path.join(model_path, input_file)
    out = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".weight_packed"):
                layer = k.rsplit(".weight_packed", 1)[0]
                out.append(f"{layer}.weight")
            elif k.endswith((".weight_scale", ".weight_shape", ".weight_zero_point")):
                continue
            else:
                out.append(k)
    return out


def _sanitize_config_for_bf16(output_path: str, text_only: bool = False) -> None:
    """Remove quantization_config from config.json for BF16 models."""
    cfg_path = os.path.join(output_path, "config.json")
    if not os.path.exists(cfg_path):
        return
    
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return
    
    changed = False
    if isinstance(cfg, dict):
        # Remove top-level quantization_config
        if "quantization_config" in cfg:
            cfg.pop("quantization_config", None)
            changed = True
        # Remove nested text_config.quantization_config
        tc = cfg.get("text_config")
        if isinstance(tc, dict) and "quantization_config" in tc:
            tc.pop("quantization_config", None)
            changed = True
        
        # Remove vision-related config if text_only
        if text_only:
            vision_config_keys = [
                "vision_config", "mm_vision_tower", "vision_tower_aux_list",
                "vision_feature_layer", "mm_projector_type",
            ]
            for key in vision_config_keys:
                if key in cfg:
                    cfg.pop(key, None)
                    changed = True
            
            # Update architecture to text-only variant
            if "architectures" in cfg:
                old_archs = cfg["architectures"]
                new_archs = [
                    arch.replace("ForConditionalGeneration", "ForCausalLM")
                    for arch in old_archs
                ]
                if new_archs != old_archs:
                    cfg["architectures"] = new_archs
                    changed = True
    
    if changed:
        tmp = cfg_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        os.replace(tmp, cfg_path)


def unpack_int4_to_bf16_gpu(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: List[int],
    weight_zero_point: Optional[torch.Tensor] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """
    GPU-accelerated unpacking of INT4 packed weights to BF16.
    
    Args:
        weight_packed: Packed INT4 weights (int32 tensor)
        weight_scale: Quantization scales
        weight_shape: Original weight shape [out_features, in_features]
        weight_zero_point: Optional zero-point values
        group_size: Group size for quantization
        
    Returns:
        Decompressed BF16 weight tensor
    """
    device = weight_packed.device
    out_features, in_features = weight_shape
    
    packed_flat = weight_packed.view(-1).to(torch.int32)
    
    # Vectorized unpacking on GPU: extract 8 nibbles from each int32
    shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
    packed_expanded = packed_flat.unsqueeze(1)
    int4_values = ((packed_expanded >> shifts) & 0xF).to(torch.int8)
    
    # Convert unsigned 4-bit to signed (two's complement)
    int4_values = torch.where(int4_values >= 8, int4_values - 16, int4_values)
    
    # Flatten and reshape
    int4_flat = int4_values.view(-1)
    total_elements = out_features * in_features
    int4_flat = int4_flat[:total_elements]
    int4_weight = int4_flat.view(out_features, in_features).to(torch.float32)
    
    # Apply dequantization: (int4 - zero_point) * scale
    if weight_zero_point is not None:
        zp = weight_zero_point.view(-1, 1).to(torch.float32)
        int4_weight = int4_weight - zp
    
    # Apply scale
    scale = weight_scale.to(torch.float32)
    
    if scale.dim() == 1:
        # Per-row scaling
        scale = scale.view(-1, 1)
        bf16_weight = (int4_weight * scale).to(torch.bfloat16)
    elif scale.dim() == 2:
        # Per-group scaling
        num_groups = scale.shape[1]
        actual_group_size = in_features // num_groups
        scale_expanded = scale.repeat_interleave(actual_group_size, dim=1)
        if scale_expanded.shape[1] < in_features:
            pad_size = in_features - scale_expanded.shape[1]
            scale_expanded = torch.cat(
                [scale_expanded, scale_expanded[:, -1:].repeat(1, pad_size)], dim=1
            )
        scale_expanded = scale_expanded[:, :in_features]
        bf16_weight = (int4_weight * scale_expanded).to(torch.bfloat16)
    else:
        bf16_weight = (int4_weight * scale.view(-1, 1)).to(torch.bfloat16)
    
    return bf16_weight


def decompress_file(
    file_name: str, model_path: str, weight_map: Dict[str, str], device: str, text_only: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Process a single safetensor file and return decompressed weights.
    
    Args:
        file_name: Input safetensors file name
        model_path: Path to input model directory
        weight_map: Weight-to-file mapping from index.json
        device: CUDA device for processing
        text_only: If True, skip vision-related tensors
        
    Returns:
        Dictionary of decompressed tensors
    """
    file_path = os.path.join(model_path, file_name)
    results = {}
    processed_layers = set()
    
    with safe_open(file_path, framework="pt", device=device) as f:
        keys = list(f.keys())
        
        for weight_name in keys:
            # Skip vision tensors if text_only mode
            if text_only and not _is_text_only_key(weight_name):
                continue
            
            tensor = f.get_tensor(weight_name)
            
            if weight_name.endswith(".weight_packed"):
                layer_name = weight_name.rsplit(".weight_packed", 1)[0]
                
                if layer_name in processed_layers:
                    continue
                
                # Skip if already decompressed
                if tensor.dtype != torch.int32:
                    output_name = f"{layer_name}.weight"
                    results[output_name] = tensor.cpu()
                else:
                    # Get associated tensors
                    scale_name = f"{layer_name}.weight_scale"
                    shape_name = f"{layer_name}.weight_shape"
                    zp_name = f"{layer_name}.weight_zero_point"
                    
                    weight_scale = None
                    weight_shape = None
                    weight_zp = None
                    
                    # Try to load from current file
                    if scale_name in keys:
                        weight_scale = f.get_tensor(scale_name)
                    if shape_name in keys:
                        shape_tensor = f.get_tensor(shape_name)
                        weight_shape = [int(x) for x in shape_tensor.tolist()]
                    if zp_name in keys:
                        weight_zp = f.get_tensor(zp_name)
                    
                    # Load from other files if needed
                    if weight_scale is None and scale_name in weight_map:
                        scale_file = weight_map[scale_name]
                        with safe_open(
                            os.path.join(model_path, scale_file),
                            framework="pt",
                            device=device,
                        ) as sf:
                            weight_scale = sf.get_tensor(scale_name)
                    
                    if weight_shape is None and shape_name in weight_map:
                        shape_file = weight_map[shape_name]
                        with safe_open(
                            os.path.join(model_path, shape_file),
                            framework="pt",
                            device=device,
                        ) as sf:
                            shape_tensor = sf.get_tensor(shape_name)
                            weight_shape = [int(x) for x in shape_tensor.tolist()]
                    
                    if weight_scale is None or weight_shape is None:
                        print(f"  Warning: Missing scale/shape for {layer_name}")
                        continue
                    
                    # GPU decompression
                    bf16_weight = unpack_int4_to_bf16_gpu(
                        tensor, weight_scale, weight_shape, weight_zp
                    )
                    
                    output_name = f"{layer_name}.weight"
                    results[output_name] = bf16_weight.cpu()
                    
                    # Free GPU memory
                    del bf16_weight
                
                processed_layers.add(layer_name)
                
            elif weight_name.endswith(
                (".weight_scale", ".weight_shape", ".weight_zero_point")
            ):
                # Skip quantization metadata
                continue
            else:
                # Copy non-quantized tensors as-is
                results[weight_name] = tensor.cpu()
            
            del tensor
    
    # Force GPU memory cleanup
    torch.cuda.empty_cache()
    
    return results


def decompress_model_incremental(
    model_path: str,
    output_path: str,
    num_gpus: int = 4,
    fresh: bool = False,
    text_only: bool = False,
    verbose: bool = True,
) -> None:
    """
    Decompress INT4 model to BF16 with incremental saving to avoid OOM.
    
    Processes files in batches, saves immediately, frees memory.
    Supports resuming from previous partial run.
    
    Args:
        model_path: Path to input INT4 model
        output_path: Path to output BF16 model
        num_gpus: Number of GPUs to use for parallel processing
        fresh: If True, remove existing output and start fresh
        text_only: If True, extract only text model (exclude vision components)
        verbose: Print progress messages
    """
    if verbose:
        print(f"=== INT4 → BF16 Streaming Decompression ===")
        print(f"Input:  {model_path}")
        print(f"Output: {output_path}")
        print(f"GPUs:   {num_gpus}")
        print(f"Mode:   {'Text-only' if text_only else 'Full model'}")
        print()
    
    if fresh and os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
    
    os.makedirs(output_path, exist_ok=True)
    
    # Check if already complete
    if os.path.exists(os.path.join(output_path, "model.safetensors.index.json")) and not fresh:
        if verbose:
            print("Output already complete. Use --fresh to restart.")
        return
    
    # Load index
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    files = sorted(set(weight_map.values()))
    total_inputs = len(files)
    
    if verbose:
        print(f"Found {total_inputs} safetensor files to process\n")
    
    # Normalize legacy shard names
    for i in range(1, total_inputs + 1):
        existing = _find_existing_shard_path(output_path, i)
        if existing is None:
            continue
        desired = os.path.join(output_path, _output_shard_name(i, total_inputs))
        if os.path.abspath(existing) != os.path.abspath(desired) and not os.path.exists(desired):
            try:
                os.rename(existing, desired)
            except OSError:
                pass
    
    def _missing_indices() -> List[int]:
        missing = []
        for i in range(1, total_inputs + 1):
            desired = os.path.join(output_path, _output_shard_name(i, total_inputs))
            if os.path.exists(desired):
                continue
            if _find_existing_shard_path(output_path, i) is not None:
                continue
            missing.append(i)
        return missing
    
    missing = _missing_indices()
    if missing:
        if verbose:
            print(f"Processing {len(missing)}/{total_inputs} missing shard(s)")
    else:
        if verbose:
            print("All shards already exist. Skipping decompression.")
    
    # Process missing shards
    for shard_idx in tqdm(missing, desc="Decompressing", disable=not verbose):
        file_name = files[shard_idx - 1]
        # Round-robin GPU assignment
        device = f"cuda:{(shard_idx - 1) % num_gpus}"
        
        # Decompress
        results = decompress_file(file_name, model_path, weight_map, device, text_only=text_only)
        
        if not results:
            continue
        
        # Save shard
        shard_name = _output_shard_name(shard_idx, total_inputs)
        shard_path = os.path.join(output_path, shard_name)
        save_file(results, shard_path)
        
        # Free memory
        del results
        gc.collect()
        torch.cuda.empty_cache()
    
    # Check if complete
    missing_after = _missing_indices()
    if missing_after:
        if verbose:
            print(f"\nStill missing {len(missing_after)}/{total_inputs} shard(s)")
        return
    
    if verbose:
        print(f"\nAll {total_inputs} shard(s) complete. Writing index...")
    
    # Build index
    new_weight_map = {}
    for i, file_name in enumerate(tqdm(files, desc="Indexing", disable=not verbose)):
        shard_file = _output_shard_name(i + 1, total_inputs)
        for out_key in _expected_output_keys_for_file(model_path, file_name):
            new_weight_map[out_key] = shard_file
    
    total_size = 0
    for i in range(1, total_inputs + 1):
        p = os.path.join(output_path, _output_shard_name(i, total_inputs))
        try:
            total_size += os.path.getsize(p)
        except OSError:
            pass
    
    new_index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)
    
    # Copy config/tokenizer files
    if verbose:
        print("Copying config files...")
    
    config_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "tiktoken.model",
        "preprocessor_config.json",
        "chat_template.jinja",
    ]
    
    import shutil
    for config_file in config_files:
        src = os.path.join(model_path, config_file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_path, config_file))
    
    # Copy model code files
    for py_file in Path(model_path).glob("*.py"):
        shutil.copy(py_file, output_path)
    
    # Remove quantization_config from config.json (and vision config if text_only)
    _sanitize_config_for_bf16(output_path, text_only=text_only)
    
    if verbose:
        print()
        print(f"=== Done! ===")
        print(f"Total size: {total_size / 1e9:.2f} GB")
        print(f"Output: {output_path}")
