#!/bin/bash
# Setup vLLM in NGC PyTorch or TRT-LLM container

set -e

VENV_DIR=${VENV_DIR:-.venv}

echo "=== Setting up vLLM ==="
echo "Virtual env: $VENV_DIR"
echo ""

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv --quiet
fi

# Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    uv venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

echo "Installing vLLM nightly..."
# Install vLLM nightly with CUDA 12.9 support
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match

echo ""
echo "=== vLLM Installation Complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Then run: bash serve_nvfp4.sh"
