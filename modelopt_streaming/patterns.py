"""Weight key pattern matching for selective quantization."""

import fnmatch


def should_quantize_tensor(key: str, mlp_only: bool = True, exclude_modules: list = None) -> bool:
    """
    Determine if a tensor key should be quantized.
    
    Args:
        key: Tensor key from model weight map (e.g., "model.layers.0.mlp.down_proj.weight")
        mlp_only: If True, only quantize MLP weights; otherwise quantize all linear layers
        exclude_modules: Optional list of glob patterns for modules to exclude from quantization
        
    Returns:
        True if the tensor should be quantized, False otherwise
    """
    if not key.endswith(".weight"):
        return False
    
    # Check against exclusion list if provided (e.g., from baseten's config)
    if exclude_modules:
        for pattern in exclude_modules:
            # Support glob patterns (* wildcards)
            if fnmatch.fnmatch(key, pattern) or fnmatch.fnmatch(key, pattern + "*"):
                return False
    
    if mlp_only:
        # Only quantize MLP weights: down_proj, gate_proj, up_proj
        # Matches patterns like:
        #   *.mlp.down_proj.weight
        #   *.mlp.experts.N.down_proj.weight
        #   *.mlp.shared_experts.down_proj.weight
        if ".mlp." not in key:
            return False
        
        mlp_weight_names = ["down_proj.weight", "gate_proj.weight", "up_proj.weight"]
        return any(key.endswith(name) for name in mlp_weight_names)
    else:
        # Quantize all linear weights except excluded patterns
        exclude_patterns = [
            "embed_tokens",
            "lm_head",
            "layernorm",
            "norm",
            ".gate.",  # MoE router gates
            "router",
            "vision",
            "projector",
        ]
        return not any(pattern in key.lower() for pattern in exclude_patterns)
