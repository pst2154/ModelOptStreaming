#!/bin/bash
# Example: Quantize Kimi-K2.5 to NVFP4 using ModelOptStreaming

set -e

INPUT_DIR="/lustre/fsw/portfolios/general/users/asteiner/Kimi-K2.5-BF16"
OUTPUT_DIR="/lustre/fsw/portfolios/general/users/asteiner/Kimi-K2.5-NVFP4-STREAM"

echo "=== Quantizing Kimi-K2.5 to NVFP4 ==="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Run with pip-installed package
modelopt-streaming quantize \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --format nvfp4 \
    --mlp_only \
    --device cuda:0 \
    --resume

echo ""
echo "=== Serving with vLLM ==="
echo "vllm serve $OUTPUT_DIR \\"
echo "    --quantization compressed_tensors \\"
echo "    --served-model-name kimi-k2.5-nvfp4 \\"
echo "    -tp 4 \\"
echo "    --mm-encoder-tp-mode data \\"
echo "    --tool-call-parser kimi_k2 \\"
echo "    --reasoning-parser kimi_k2 \\"
echo "    --trust-remote-code \\"
echo "    --kv-cache-dtype fp8 \\"
echo "    --port 8000"
