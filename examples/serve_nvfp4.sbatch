#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -A general_cs_infra
#SBATCH -p batch_long
#SBATCH --exclude=nvl72d012-T03
#SBATCH --job-name=vllm-nvfp4
#SBATCH --output=serve_nvfp4_%j.log
#SBATCH --error=serve_nvfp4_%j.log

set -e

BASE_DIR=/lustre/fsw/portfolios/general/users/asteiner
MODEL_PATH=${MODEL_PATH:-${BASE_DIR}/Kimi-K2.5-NVFP4-CALIBRATED-20260201_174511}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-kimi-k2.5-nvfp4-fixed}
SERVER_PORT=${SERVER_PORT:-8000}
TP_SIZE=${TP_SIZE:-4}

echo "=== vLLM NVFP4 Serving ==="
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "Model: $MODEL_PATH"
echo "Served name: $SERVED_MODEL_NAME"
echo "Port: $SERVER_PORT"
echo "TP: $TP_SIZE"
echo ""

# Use official vLLM Docker image (nightly build with all dependencies)
srun --container-image=docker://vllm/vllm-openai:nightly \
     --container-mounts=${BASE_DIR}:${BASE_DIR} \
     bash -c "
# Redirect all cache directories to Lustre to avoid /home quota
export VLLM_COMPILE_MODE=none
export TORCHINDUCTOR_CACHE_DIR=${BASE_DIR}/.cache/torch_compile
export HF_HOME=${BASE_DIR}/.cache/huggingface
export TRANSFORMERS_CACHE=${BASE_DIR}/.cache/huggingface
export VLLM_CACHE_ROOT=${BASE_DIR}/.cache/vllm
export XDG_CACHE_HOME=${BASE_DIR}/.cache
export PIP_CACHE_DIR=${BASE_DIR}/.cache/pip

# Pin to working vLLM version (before regression bug in dev118+)
echo '=== Installing vLLM v0.16.0rc1.dev78 ==='
pip install --no-cache-dir 'vllm @ https://wheels.vllm.ai/nightly/vllm-0.16.0rc1.dev78+gb6bb2842c-cp38-abi3-manylinux_2_31_aarch64.whl'

echo ''
echo '=== Verifying vLLM version ==='
python -c 'import vllm; print(f\"vLLM version: {vllm.__version__}\")'

echo ''
echo '=== Starting vLLM server ==='
echo 'Server will be available at http://'\$(hostname)':${SERVER_PORT}'
echo 'Model: ${MODEL_PATH}'
echo 'Quantization: NVFP4 (auto-detected from config.json)'
echo 'Hardware required: SM 10.0 (Blackwell B200/B300/GB200)'
echo ''

vllm serve ${MODEL_PATH} \\
    --served-model-name ${SERVED_MODEL_NAME} \\
    -tp ${TP_SIZE} \\
    --mm-encoder-tp-mode data \\
    --tool-call-parser kimi_k2 \\
    --reasoning-parser kimi_k2 \\
    --trust-remote-code \\
    --kv-cache-dtype fp8 \\
    --port ${SERVER_PORT}
"

echo ""
echo "=== Server stopped ==="
echo "Finished at: $(date)"
