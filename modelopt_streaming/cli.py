"""Command-line interface for ModelOptStreaming."""

import argparse
import sys

from .quantizer import StreamingQuantizer
from .formats import QuantizationFormat


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ModelOptStreaming: Memory-efficient streaming quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic NVFP4 quantization (MLP weights only)
  modelopt-streaming quantize --input_dir ./model-bf16 --output_dir ./model-nvfp4

  # Quantize all linear weights
  modelopt-streaming quantize --input_dir ./model-bf16 --output_dir ./model-nvfp4 --all_linear

  # Resume interrupted quantization
  modelopt-streaming quantize --input_dir ./model-bf16 --output_dir ./model-nvfp4 --resume
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
        "--block_size",
        type=int,
        default=16,
        help="Block size for group quantization (default: 16)"
    )
    quantize_parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device for quantization (default: cuda:0)"
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
            block_size=args.block_size,
            device=args.device,
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


if __name__ == "__main__":
    main()
