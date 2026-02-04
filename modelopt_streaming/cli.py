"""Command-line interface for ModelOptStreaming."""

import argparse
import sys

from .quantizer import StreamingQuantizer
from .formats import QuantizationFormat
from .decompress import decompress_model_incremental
from .utils import (
    extract_exclude_modules,
    save_exclude_modules,
    print_exclude_modules_summary,
    extract_text_only_model,
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ModelOptStreaming: Memory-efficient streaming quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # NVFP4 quantization (MLP weights only)
  modelopt-streaming quantize --input_dir ./model-bf16 --output_dir ./model-nvfp4

  # Quantize all linear weights
  modelopt-streaming quantize --input_dir ./model-bf16 --output_dir ./model-nvfp4 --all_linear

  # INT4 decompression to BF16
  modelopt-streaming decompress --input_dir ./model-int4 --output_dir ./model-bf16
  
  # Extract exclusion patterns from a config
  modelopt-streaming extract --config ./model/hf_quant_config.json --verbose
  
  # Extract text-only model from multimodal BF16 model
  modelopt-streaming extract-text --input_dir ./model-multimodal-bf16 --output_dir ./model-text-only-bf16
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Quantize command
    quantize_parser = subparsers.add_parser("quantize", help="Quantize a model")
    quantize_parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to input BF16/FP16 model directory"
    )
    quantize_parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to output quantized model directory"
    )
    quantize_parser.add_argument(
        "--format",
        default="nvfp4",
        choices=["nvfp4", "fp8", "int4_awq"],
        help="Quantization format (default: nvfp4)"
    )
    quantize_parser.add_argument(
        "--mlp_only",
        action="store_true",
        default=True,
        help="Quantize only MLP weights (default: True)"
    )
    quantize_parser.add_argument(
        "--all_linear",
        action="store_true",
        help="Quantize all linear weights (overrides --mlp_only)"
    )
    quantize_parser.add_argument(
        "--exclude_config",
        type=str,
        help="Path to JSON config with 'exclude_modules' list (e.g., baseten's hf_quant_config.json)"
    )
    quantize_parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Block size for group quantization (default: 16)"
    )
    quantize_parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device for quantization in single-GPU mode (default: cuda:0)"
    )
    quantize_parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel quantization (default: 1)"
    )
    quantize_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output shards"
    )
    quantize_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    quantize_parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable calibration for activation quantization (slower, more accurate)"
    )
    quantize_parser.add_argument(
        "--calib_size",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)"
    )
    quantize_parser.add_argument(
        "--calib_dataset",
        default="cnn_dailymail",
        help="HuggingFace dataset for calibration (default: cnn_dailymail)"
    )
    
    # Decompress command
    decompress_parser = subparsers.add_parser("decompress", help="Decompress INT4 model to BF16")
    decompress_parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to input INT4 (compressed-tensors) model directory"
    )
    decompress_parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to output BF16 model directory"
    )
    decompress_parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help="Number of GPUs to use for decompression (default: 4)"
    )
    decompress_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Remove existing output and start fresh"
    )
    decompress_parser.add_argument(
        "--text-only",
        action="store_true",
        help="Extract text-only model (exclude vision components)"
    )
    decompress_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    # Extract command (exclusion patterns)
    extract_parser = subparsers.add_parser("extract", help="Extract exclusion patterns from config")
    extract_parser.add_argument(
        "--config",
        required=True,
        help="Path to quantization config file (hf_quant_config.json or config.json)"
    )
    extract_parser.add_argument(
        "--output",
        help="Optional output file to save patterns (one per line)"
    )
    extract_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all patterns (otherwise just summary)"
    )
    
    # Extract-text command (text-only model extraction)
    extract_text_parser = subparsers.add_parser("extract-text", help="Extract text-only model from multimodal BF16 model")
    extract_text_parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to input multimodal BF16 model directory"
    )
    extract_text_parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to output text-only BF16 model directory"
    )
    extract_text_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "quantize":
        mlp_only = not args.all_linear if args.all_linear else args.mlp_only
        
        quantizer = StreamingQuantizer(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            format=args.format,
            mlp_only=mlp_only,
            exclude_config=args.exclude_config,
            block_size=args.block_size,
            device=args.device,
            num_gpus=args.num_gpus,
            resume=args.resume,
            verbose=not args.quiet,
            calibrate=args.calibrate,
            calib_size=args.calib_size,
            calib_dataset=args.calib_dataset,
        )
        
        try:
            quantizer.run()
        except Exception as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == "decompress":
        try:
            decompress_model_incremental(
                model_path=args.input_dir,
                output_path=args.output_dir,
                num_gpus=args.num_gpus,
                fresh=args.fresh,
                text_only=args.text_only,
                verbose=not args.quiet,
            )
        except Exception as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == "extract":
        try:
            exclude_modules = extract_exclude_modules(args.config)
            print_exclude_modules_summary(exclude_modules, verbose=args.verbose)
            
            if args.output:
                save_exclude_modules(exclude_modules, args.output)
                print(f"Saved patterns to: {args.output}")
        except Exception as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == "extract-text":
        try:
            extract_text_only_model(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                verbose=not args.quiet,
            )
        except Exception as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
