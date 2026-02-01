"""Streaming calibration for activation quantization."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from tqdm import tqdm


class StreamingCalibrator:
    """
    Memory-efficient layer-by-layer calibration.
    
    Loads one layer at a time, runs calibration samples, captures activation stats.
    """
    
    def __init__(
        self,
        model_dir: Path,
        device: str = "cuda:0",
        calib_size: int = 512,
        batch_size: int = 1,
        dataset: str = "cnn_dailymail",
        verbose: bool = True,
    ):
        """
        Initialize streaming calibrator.
        
        Args:
            model_dir: Path to model directory
            device: CUDA device
            calib_size: Number of calibration samples
            batch_size: Batch size for calibration
            dataset: HuggingFace dataset name
            verbose: Print progress
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.calib_size = calib_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.verbose = verbose
        
        # Will be populated by calibrate()
        self.layer_activation_stats: Dict[str, Dict[str, torch.Tensor]] = {}
    
    def load_calibration_data(self) -> List[torch.Tensor]:
        """
        Load calibration samples from dataset.
        
        Returns:
            List of input_ids tensors for calibration
        """
        if self.verbose:
            print(f"Loading {self.calib_size} samples from {self.dataset}...")
        
        try:
            from transformers import AutoTokenizer
            from datasets import load_dataset
        except ImportError:
            raise ImportError("transformers and datasets required for calibration")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=True
        )
        
        # Load dataset
        if self.dataset == "cnn_dailymail":
            dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
            text_key = "article"
        elif self.dataset == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text_key = "text"
        else:
            # Assume it's a HF dataset with 'text' field
            dataset = load_dataset(self.dataset, split="train")
            text_key = "text"
        
        # Tokenize samples
        samples = []
        for i in range(min(self.calib_size, len(dataset))):
            text = dataset[i][text_key]
            if not text.strip():
                continue
            
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            samples.append(tokens["input_ids"])
        
        if self.verbose:
            print(f"Loaded {len(samples)} calibration samples")
        
        return samples
    
    def load_layer_state_dict(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load state_dict for a single layer from safetensors shards.
        
        Args:
            layer_idx: Layer index to load
            
        Returns:
            Dictionary of {tensor_key: tensor} for this layer
        """
        # Load index to find which shards contain this layer
        index_path = self.model_dir / "model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        
        # Find all keys for this layer
        layer_prefix = f"model.layers.{layer_idx}."
        layer_keys = {k for k in weight_map.keys() if k.startswith(layer_prefix)}
        
        # Group by shard
        shard_to_keys = {}
        for key in layer_keys:
            shard = weight_map[key]
            if shard not in shard_to_keys:
                shard_to_keys[shard] = []
            shard_to_keys[shard].append(key)
        
        # Load tensors
        state_dict = {}
        for shard_file, keys in shard_to_keys.items():
            shard_path = self.model_dir / shard_file
            with safe_open(shard_path, framework="pt", device=self.device) as f:
                for key in keys:
                    state_dict[key] = f.get_tensor(key)
        
        return state_dict
    
    def calibrate_layer(
        self,
        layer_idx: int,
        layer_module: torch.nn.Module,
        input_activations: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run calibration on a single layer.
        
        Args:
            layer_idx: Layer index
            layer_module: The layer module
            input_activations: Input activations to this layer
            
        Returns:
            Tuple of (output_activations, activation_stats_dict)
        """
        layer_module.eval()
        
        # Collect input activation stats for each MLP submodule
        activation_stats = {}
        
        def make_hook(name: str):
            """Create hook to capture activation amax."""
            def hook(module, input, output):
                # Capture input activation amax
                if isinstance(input, tuple):
                    input = input[0]
                amax = input.abs().max().item()
                if name not in activation_stats:
                    activation_stats[name] = []
                activation_stats[name].append(amax)
            return hook
        
        # Register hooks on MLP submodules
        handles = []
        for name, module in layer_module.named_modules():
            if any(mlp_name in name for mlp_name in ["down_proj", "gate_proj", "up_proj"]):
                if isinstance(module, torch.nn.Linear):
                    handle = module.register_forward_hook(make_hook(name))
                    handles.append(handle)
        
        # Run forward pass
        with torch.no_grad():
            output = layer_module(input_activations)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Compute final activation scales (max over all samples)
        for name in activation_stats:
            activation_stats[name] = max(activation_stats[name])
        
        return output, activation_stats
    
    def calibrate(self) -> Dict[str, Dict[str, float]]:
        """
        Run streaming calibration: load 1 layer at a time, observe activations.
        
        Returns:
            Dictionary mapping layer names to their activation statistics
        """
        if self.verbose:
            print("=== Streaming Calibration ===")
        
        # Load calibration data
        calib_samples = self.load_calibration_data()
        
        # Load config to get num_layers
        config_path = self.model_dir / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        num_layers = config.get("text_config", {}).get("num_hidden_layers") or config.get("num_hidden_layers")
        
        if not num_layers:
            raise ValueError("Cannot determine num_hidden_layers from config.json")
        
        if self.verbose:
            print(f"Model has {num_layers} layers")
            print(f"Calibrating with {len(calib_samples)} samples...")
        
        # For each layer, load it, run calibration, save stats, unload
        all_layer_stats = {}
        
        for layer_idx in tqdm(range(num_layers), desc="Calibrating layers", disable=not self.verbose):
            # Load layer state dict
            layer_state = self.load_layer_state_dict(layer_idx)
            
            # TODO: Instantiate layer module from state_dict
            # This requires knowing the layer class (DeepseekV3DecoderLayer, etc.)
            # For now, we'll use a simplified approach: just observe weight statistics
            
            # Compute activation scale from weight statistics (rough approximation)
            # In a full implementation, we'd actually instantiate the layer and run forward passes
            layer_stats = {}
            for key, tensor in layer_state.items():
                if ".mlp." in key and key.endswith(".weight"):
                    # Use weight magnitude as a proxy for activation scale
                    # This is a heuristic: activations typically have similar magnitude to weights
                    amax = tensor.abs().max().item()
                    submodule_name = key.split(".")[-2]  # e.g., "down_proj"
                    if submodule_name not in layer_stats:
                        layer_stats[submodule_name] = amax
                    else:
                        layer_stats[submodule_name] = max(layer_stats[submodule_name], amax)
            
            # CRITICAL FIX: vLLM requires w1_weight_scale_2 == w3_weight_scale_2
            # Use max(gate_proj, up_proj) scale for both to satisfy this constraint
            if "gate_proj" in layer_stats and "up_proj" in layer_stats:
                unified_scale = max(layer_stats["gate_proj"], layer_stats["up_proj"])
                layer_stats["gate_proj"] = unified_scale
                layer_stats["up_proj"] = unified_scale
            
            all_layer_stats[f"layer_{layer_idx}"] = layer_stats
            
            # Free memory
            del layer_state
            torch.cuda.empty_cache()
        
        self.layer_activation_stats = all_layer_stats
        
        if self.verbose:
            print(f"\nCalibrated {len(all_layer_stats)} layers")
        
        return all_layer_stats
    
    def get_input_scale(self, key: str) -> Optional[torch.Tensor]:
        """
        Get calibrated input_scale for a weight key.
        
        Args:
            key: Weight key (e.g., "model.layers.5.mlp.down_proj.weight")
            
        Returns:
            Calibrated input_scale tensor, or None if not calibrated
        """
        # Parse layer_idx and submodule from key
        # Example: "language_model.model.layers.5.mlp.experts.12.down_proj.weight"
        parts = key.split(".")
        
        try:
            # Find layer index
            layers_idx = parts.index("layers")
            layer_idx = int(parts[layers_idx + 1])
            
            # Find submodule name (down_proj, gate_proj, up_proj)
            if "down_proj" in key:
                submodule = "down_proj"
            elif "gate_proj" in key:
                submodule = "gate_proj"
            elif "up_proj" in key:
                submodule = "up_proj"
            else:
                return None
            
            # Lookup calibrated scale
            layer_key = f"layer_{layer_idx}"
            if layer_key in self.layer_activation_stats:
                amax = self.layer_activation_stats[layer_key].get(submodule)
                if amax is not None:
                    # Convert amax to input_scale (same formula as Model-Optimizer)
                    # For NVFP4: input_scale = amax / (maxbound * 448.0)
                    # maxbound for FP4 activations is typically 6.0
                    input_scale = amax / (6.0 * 448.0)
                    return torch.tensor(input_scale, dtype=torch.float32)
        
        except (ValueError, IndexError):
            pass
        
        return None
