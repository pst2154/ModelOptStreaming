"""Core streaming quantization logic."""

import json
import shutil
from pathlib import Path
from typing import Dict, Set, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from .formats import QuantizationFormat, get_quantization_config
from .patterns import should_quantize_tensor
from .nvfp4 import quantize_weight_nvfp4, compute_dummy_input_scale
from .calibration import StreamingCalibrator


class StreamingQuantizer:
    """
    Memory-efficient streaming quantizer for large language models.
    
    Processes safetensors shards one at a time to avoid loading the full model.
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        format: str = "nvfp4",
        mlp_only: bool = True,
        block_size: int = 16,
        device: str = "cuda:0",
        resume: bool = False,
        verbose: bool = True,
        calibrate: bool = False,
        calib_size: int = 512,
        calib_dataset: str = "cnn_dailymail",
    ):
        """
        Initialize streaming quantizer.
        
        Args:
            input_dir: Path to input BF16/FP16 model directory
            output_dir: Path to output quantized model directory
            format: Quantization format ('nvfp4', 'fp8', 'int4_awq')
            mlp_only: Quantize only MLP weights (faster, safer)
            block_size: Block size for group quantization (default: 16)
            device: CUDA device for quantization kernels
            resume: Resume from existing output shards
            verbose: Print progress information
            calibrate: Enable calibration for activation quantization (W4A4)
            calib_size: Number of calibration samples
            calib_dataset: HuggingFace dataset for calibration
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.format = QuantizationFormat(format)
        self.mlp_only = mlp_only
        self.block_size = block_size
        self.device = device
        self.resume = resume
        self.verbose = verbose
        self.calibrate = calibrate
        self.calib_size = calib_size
        self.calib_dataset = calib_dataset
        
        # Calibration state
        self.calibrator: Optional[StreamingCalibrator] = None
        
        # Validate inputs
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        index_path = self.input_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Load index
        with open(index_path, "r") as f:
            self.input_index = json.load(f)
        
        self.weight_map = self.input_index["weight_map"]
        
        # Setup Model-Optimizer
        self._setup_backend()
        
    def _setup_backend(self):
        """Import and configure quantization backend."""
        if self.format == QuantizationFormat.NVFP4:
            try:
                from modelopt.torch.quantization.qtensor import NVFP4QTensor
                self.quantizer_class = NVFP4QTensor
            except ImportError:
                raise ImportError(
                    "modelopt not installed. Install via:\n"
                    "  pip install git+https://github.com/NVIDIA/Model-Optimizer.git@zhiyu/support-kimi-k2.5-ptq"
                )
        else:
            raise NotImplementedError(f"Format {self.format} not yet implemented")
    
    def get_keys_to_quantize(self) -> Set[str]:
        """Determine which weight keys should be quantized."""
        return {
            key for key in self.weight_map.keys()
            if should_quantize_tensor(key, mlp_only=self.mlp_only)
        }
    
    def process_shard(
        self,
        input_path: Path,
        output_path: Path,
        keys_to_quantize: Set[str],
    ) -> Dict[str, tuple]:
        """
        Process a single safetensors shard.
        
        Args:
            input_path: Path to input shard
            output_path: Path to output shard
            keys_to_quantize: Set of tensor keys to quantize
            
        Returns:
            Dictionary mapping tensor keys to their output shapes
        """
        output_tensors = {}
        
        with safe_open(input_path, framework="pt", device="cpu") as f:
            tensor_keys = f.keys()
            
            for key in tqdm(
                tensor_keys,
                desc=f"Processing {input_path.name}",
                leave=False,
                disable=not self.verbose
            ):
                if key in keys_to_quantize:
                    # Quantize this weight
                    weight = f.get_tensor(key)
                    
                    try:
                        if self.format == QuantizationFormat.NVFP4:
                            packed_weight, weight_scale, weight_scale_2 = quantize_weight_nvfp4(
                                weight,
                                self.quantizer_class,
                                self.block_size,
                                self.device
                            )
                            
                            # Store quantized weight and scales
                            output_tensors[key] = packed_weight
                            output_tensors[f"{key}_scale"] = weight_scale
                            output_tensors[f"{key}_scale_2"] = weight_scale_2
                            
                            # Add input_scale (calibrated if available, else dummy)
                            if self.calibrator:
                                input_scale = self.calibrator.get_input_scale(key)
                            else:
                                input_scale = None
                            
                            if input_scale is None:
                                input_scale = compute_dummy_input_scale(weight.shape)
                            
                            output_tensors[key.replace(".weight", ".input_scale")] = input_scale
                        
                        del weight
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"\nERROR quantizing {key}: {e}")
                        # Fallback: copy original weight
                        output_tensors[key] = f.get_tensor(key)
                else:
                    # Copy tensor as-is
                    output_tensors[key] = f.get_tensor(key)
        
        # Save output shard
        save_file(output_tensors, output_path)
        return {k: v.shape for k, v in output_tensors.items()}
    
    def build_index(self, shard_metadata: Dict[str, Dict]) -> None:
        """Build model.safetensors.index.json from shard metadata."""
        weight_map = {}
        total_size = 0
        
        for shard_name, tensors in shard_metadata.items():
            for key, shape in tensors.items():
                weight_map[key] = shard_name
                # Estimate size
                numel = 1
                for dim in shape:
                    numel *= dim
                # Rough dtype sizes
                if key.endswith("_scale_2") or key.endswith(".input_scale"):
                    total_size += numel * 4  # float32
                elif key.endswith("_scale"):
                    total_size += numel * 1  # float8
                else:
                    total_size += numel * 1  # uint8
        
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map
        }
        
        with open(self.output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
    
    def copy_metadata_files(self) -> None:
        """Copy config and tokenizer files to output directory."""
        metadata_files = [
            "config.json",
            "generation_config.json",
            "tiktoken.model",
            "preprocessor_config.json",
            "chat_template.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        
        for filename in metadata_files:
            src = self.input_dir / filename
            if src.exists():
                shutil.copy2(src, self.output_dir / filename)
        
        # Copy model code files
        for src_file in self.input_dir.glob("*.py"):
            shutil.copy2(src_file, self.output_dir / src_file.name)
    
    def inject_quantization_config(self) -> None:
        """Inject quantization_config into config.json."""
        config_path = self.output_dir / "config.json"
        if not config_path.exists():
            if self.verbose:
                print("WARNING: config.json not found, skipping quantization_config injection")
            return
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        quant_config = get_quantization_config(
            self.format,
            group_size=self.block_size,
            mlp_only=self.mlp_only
        )
        
        config["quantization_config"] = quant_config
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Also save standalone hf_quant_config.json for backward compatibility
        with open(self.output_dir / "hf_quant_config.json", "w") as f:
            json.dump({
                "producer": quant_config["producer"],
                "quantization": quant_config["quantization"]
            }, f, indent=4)
    
    def run(self) -> None:
        """Execute the streaming quantization pipeline."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run calibration if requested
        if self.calibrate:
            if self.verbose:
                print(f"=== Calibration Enabled ===")
                print(f"Samples: {self.calib_size}")
                print(f"Dataset: {self.calib_dataset}")
                print()
            
            self.calibrator = StreamingCalibrator(
                model_dir=self.input_dir,
                device=self.device,
                calib_size=self.calib_size,
                batch_size=1,
                dataset=self.calib_dataset,
                verbose=self.verbose,
            )
            self.calibrator.calibrate()
        
        # Determine keys to quantize
        keys_to_quantize = self.get_keys_to_quantize()
        
        if self.verbose:
            print(f"=== ModelOptStreaming v0.1.0 ===")
            print(f"Input:  {self.input_dir}")
            print(f"Output: {self.output_dir}")
            print(f"Format: {self.format.value}")
            print(f"Mode: {'MLP-only' if self.mlp_only else 'All Linear'}")
            print(f"Calibration: {'Enabled' if self.calibrate else 'Disabled (weight-only)'}")
            print(f"Total keys: {len(self.weight_map)}")
            print(f"Keys to quantize: {len(keys_to_quantize)}")
            print(f"Device: {self.device}")
            print()
        
        # Get unique shard files
        shard_files = sorted(set(self.weight_map.values()))
        
        # Resume detection
        completed_shards = set()
        if self.resume:
            for shard_file in shard_files:
                if (self.output_dir / shard_file).exists():
                    completed_shards.add(shard_file)
            if completed_shards and self.verbose:
                print(f"Resume: Found {len(completed_shards)} existing shards, skipping them.")
        
        # Process each shard
        shard_metadata = {}
        
        for shard_file in tqdm(
            shard_files,
            desc="Processing shards",
            disable=not self.verbose
        ):
            output_shard_path = self.output_dir / shard_file
            
            if shard_file in completed_shards:
                # Load metadata from existing shard
                with safe_open(output_shard_path, framework="pt", device="cpu") as f:
                    shard_metadata[shard_file] = {k: f.get_tensor(k).shape for k in f.keys()}
                continue
            
            input_shard_path = self.input_dir / shard_file
            
            if not input_shard_path.exists():
                if self.verbose:
                    print(f"\nWARNING: Input shard not found: {input_shard_path}")
                continue
            
            # Process shard
            shard_meta = self.process_shard(
                input_shard_path,
                output_shard_path,
                keys_to_quantize,
            )
            shard_metadata[shard_file] = shard_meta
            
            # Free memory
            torch.cuda.empty_cache()
        
        # Build index
        if self.verbose:
            print("\nBuilding index.json...")
        self.build_index(shard_metadata)
        
        # Copy metadata
        if self.verbose:
            print("Copying config files...")
        self.copy_metadata_files()
        
        # Inject quantization config
        if self.verbose:
            print("Creating quantization_config...")
        self.inject_quantization_config()
        
        if self.verbose:
            print("\n=== Done! ===")
            print(f"Quantized model saved to: {self.output_dir}")
            print(f"\nTo serve with vLLM:")
            print(f"  vllm serve {self.output_dir} --quantization compressed_tensors -tp 4")
