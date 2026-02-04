"""Utility functions for ModelOptStreaming."""

import json
import shutil
from pathlib import Path
from typing import List, Optional, Set

from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def extract_exclude_modules(config_path: str) -> List[str]:
    """
    Extract exclude_modules list from a quantization config file.
    
    Supports multiple config formats:
    - Baseten format: .quantization.exclude_modules
    - Top-level format: .exclude_modules
    - HF format: config.json with quantization_config.exclude_modules
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        List of exclusion patterns (glob patterns with wildcards)
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If no exclude_modules found in config
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    exclude_modules = None
    
    # Try different config formats
    if "quantization" in config and isinstance(config["quantization"], dict):
        if "exclude_modules" in config["quantization"]:
            exclude_modules = config["quantization"]["exclude_modules"]
    
    if exclude_modules is None and "quantization_config" in config:
        if "exclude_modules" in config["quantization_config"]:
            exclude_modules = config["quantization_config"]["exclude_modules"]
    
    if exclude_modules is None and "exclude_modules" in config:
        exclude_modules = config["exclude_modules"]
    
    if exclude_modules is None:
        raise ValueError(f"No 'exclude_modules' found in config: {config_path}")
    
    return exclude_modules


def save_exclude_modules(exclude_modules: List[str], output_path: str) -> None:
    """
    Save exclude_modules list to a text file (one pattern per line).
    
    Args:
        exclude_modules: List of exclusion patterns
        output_path: Path to output text file
    """
    output_path = Path(output_path)
    
    with open(output_path, "w") as f:
        for pattern in exclude_modules:
            f.write(f"{pattern}\n")


def print_exclude_modules_summary(exclude_modules: List[str], verbose: bool = False) -> None:
    """
    Print a summary of exclusion patterns.
    
    Args:
        exclude_modules: List of exclusion patterns
        verbose: If True, print all patterns; otherwise just summary stats
    """
    print(f"Total exclusion patterns: {len(exclude_modules)}")
    print()
    
    # Categorize patterns
    categories = {
        "lm_head": [],
        "embed_tokens": [],
        "layer_specific": [],
        "attention": [],
        "mlp": [],
        "other": []
    }
    
    for pattern in exclude_modules:
        if "lm_head" in pattern:
            categories["lm_head"].append(pattern)
        elif "embed_tokens" in pattern:
            categories["embed_tokens"].append(pattern)
        elif "self_attn" in pattern or "attention" in pattern:
            categories["attention"].append(pattern)
        elif "mlp" in pattern:
            categories["mlp"].append(pattern)
        elif "model.layers." in pattern:
            categories["layer_specific"].append(pattern)
        else:
            categories["other"].append(pattern)
    
    print("Pattern breakdown:")
    for category, patterns in categories.items():
        if patterns:
            print(f"  {category}: {len(patterns)} patterns")
    print()
    
    if verbose:
        print("All patterns:")
        for i, pattern in enumerate(exclude_modules, 1):
            print(f"  {i:3d}. {pattern}")
        print()


def is_text_only_key(key: str) -> bool:
    """
    Determine if a tensor key belongs to the text-only model.
    
    Args:
        key: Tensor key from model weight map
        
    Returns:
        True if the tensor is part of the text-only model, False if it's vision-related
    """
    # Exclude vision-related keys
    vision_patterns = [
        "vision_model",
        "vision_projection",
        "vision_tower",
        "mm_projector",
        "visual",
        "image_encoder",
        "vision_embed_tokens",
    ]
    
    key_lower = key.lower()
    return not any(pattern in key_lower for pattern in vision_patterns)


def extract_text_only_model(
    input_dir: str,
    output_dir: str,
    verbose: bool = True
) -> None:
    """
    Extract text-only model from a multimodal BF16 model.
    
    Copies only text-related layers (model.layers.*, lm_head, embed_tokens, etc.)
    and excludes vision components (vision_model, vision_projection, etc.).
    
    Args:
        input_dir: Path to input BF16 model directory
        output_dir: Path to output text-only model directory
        verbose: Print progress information
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load index
    index_path = input_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    with open(index_path, "r") as f:
        input_index = json.load(f)
    
    weight_map = input_index["weight_map"]
    
    # Filter for text-only keys
    text_only_keys = {k: v for k, v in weight_map.items() if is_text_only_key(k)}
    
    if verbose:
        print(f"=== Extract Text-Only Model ===")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Total keys: {len(weight_map)}")
        print(f"Text-only keys: {len(text_only_keys)}")
        print(f"Vision keys removed: {len(weight_map) - len(text_only_keys)}")
        print()
    
    # Get unique shards needed
    input_shards_needed = sorted(set(text_only_keys.values()))
    
    # Process shards
    output_weight_map = {}
    shard_counter = 1
    
    for input_shard in tqdm(input_shards_needed, desc="Processing shards", disable=not verbose):
        input_shard_path = input_dir / input_shard
        
        if not input_shard_path.exists():
            if verbose:
                print(f"WARNING: Shard not found: {input_shard_path}")
            continue
        
        # Determine which keys to extract from this shard
        keys_in_shard = [k for k, v in text_only_keys.items() if v == input_shard]
        
        if not keys_in_shard:
            continue
        
        # Load and extract text-only tensors
        output_tensors = {}
        with safe_open(input_shard_path, framework="pt", device="cpu") as f:
            for key in keys_in_shard:
                output_tensors[key] = f.get_tensor(key)
        
        # Save output shard
        output_shard = f"model-{shard_counter:05d}-of-{len(input_shards_needed):05d}.safetensors"
        output_shard_path = output_dir / output_shard
        save_file(output_tensors, output_shard_path)
        
        # Update weight map
        for key in keys_in_shard:
            output_weight_map[key] = output_shard
        
        shard_counter += 1
    
    # Build output index
    output_index = {
        "metadata": {"total_size": input_index["metadata"].get("total_size", 0)},
        "weight_map": output_weight_map
    }
    
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(output_index, f, indent=2)
    
    # Copy and update config files
    config_path = input_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Remove vision-related config keys
        vision_config_keys = [
            "vision_config",
            "mm_vision_tower",
            "vision_tower_aux_list",
            "vision_feature_layer",
            "mm_projector_type",
        ]
        
        for key in vision_config_keys:
            config.pop(key, None)
        
        # Update architecture to text-only
        if "architectures" in config:
            # Update architecture name to text-only variant
            config["architectures"] = [
                arch.replace("ForConditionalGeneration", "ForCausalLM")
                for arch in config["architectures"]
            ]
        
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    # Copy other metadata files
    metadata_files = [
        "generation_config.json",
        "tiktoken.model",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    
    for filename in metadata_files:
        src = input_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
    
    # Copy Python files
    for src_file in input_dir.glob("*.py"):
        shutil.copy2(src_file, output_dir / src_file.name)
    
    if verbose:
        print("\n=== Done! ===")
        print(f"Text-only model saved to: {output_dir}")
        print(f"Shards: {shard_counter - 1}")
        print(f"Keys: {len(output_weight_map)}")
