#!/bin/bash
# Serve NVFP4-quantized model with vLLM

set -e

MODEL_PATH=${MODEL_PATH:-/lustre/fsw/portfolios/general/users/asteiner/Kimi-K2.5-NVFP4-STREAM}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-kimi-k2.5-nvfp4}
SERVER_PORT=${SERVER_PORT:-8000}
TP_SIZE=${TP_SIZE:-4}

echo "=== vLLM NVFP4 Model Server ==="
echo "Model: $MODEL_PATH"
echo "Name: $SERVED_MODEL_NAME"
echo "Port: $SERVER_PORT"
echo "TP: $TP_SIZE"
echo ""

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "ERROR: vLLM not found. Run setup_vllm.sh first."
    exit 1
fi

# Serve the model
# vLLM auto-detects NVFP4 from quantization_config in config.json
vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    -tp "$TP_SIZE" \
    --mm-encoder-tp-mode data \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --port "$SERVER_PORT"

# If auto-detection fails, add: --quantization compressed_tensors
