# Examples

## Kimi-K2.5 NVFP4 Quantization

```bash
bash kimi_k25_nvfp4.sh
```

Converts Kimi-K2.5 (684B parameters) to NVFP4 format in ~12 minutes using <100GB RAM.

## Custom Quantization

```python
from modelopt_streaming import StreamingQuantizer

# Basic usage
quantizer = StreamingQuantizer(
    input_dir="/path/to/model-bf16",
    output_dir="/path/to/model-nvfp4",
    format="nvfp4",
    mlp_only=True,
    device="cuda:0"
)
quantizer.run()

# Quantize all linear weights
quantizer = StreamingQuantizer(
    input_dir="/path/to/model-bf16",
    output_dir="/path/to/model-nvfp4",
    format="nvfp4",
    mlp_only=False,  # Quantize attention + MLP
    device="cuda:0"
)
quantizer.run()
```

## Serving with vLLM

### Setup (one-time)

```bash
# In NGC PyTorch container
bash setup_vllm.sh
source .venv/bin/activate
```

### Serve the model

```bash
export MODEL_PATH=/path/to/model-nvfp4
export SERVED_MODEL_NAME=kimi-k2.5-nvfp4
export SERVER_PORT=8000

bash serve_nvfp4.sh
```

### SLURM batch serving

```bash
sbatch serve_nvfp4.sbatch
```

### Test the server

```bash
python test_serving.py
```

See [serve_README.md](serve_README.md) for detailed setup and troubleshooting.

## Serving with TensorRT-LLM

```bash
# Convert to TensorRT-LLM engine
trtllm-build \
    --checkpoint_dir /path/to/model-nvfp4 \
    --output_dir /path/to/engine \
    --gemm_plugin auto \
    --gpt_attention_plugin auto \
    --tp_size 4
```
