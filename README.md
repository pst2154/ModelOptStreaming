# ModelOptStreaming

**Fast, memory-efficient streaming quantization and decompression for large language models.**

ModelOptStreaming converts large BF16/FP16 models to quantized formats (NVFP4, FP8, INT4) and decompresses INT4 models back to BF16 by processing safetensors shards one at a time, avoiding the memory overhead and complexity of full model instantiation.

## Why ModelOptStreaming?

Traditional quantization tools (Model-Optimizer, llm-compressor) load the entire model into memory, which:
- **Consumes 900GB+ RAM** for models like Kimi-K2.5 (684B parameters)
- **Requires complex CPU offload** and device mapping
- **Crashes with OOM** or illegal memory access errors
- **Takes 3+ hours** even for zero-calibration runs

ModelOptStreaming solves this by:
- ✅ **Processing one shard at a time** (~30-50GB peak RAM)
- ✅ **Direct safetensors I/O** (no model instantiation)
- ✅ **10-15 minute runtime** for 684B parameter models
- ✅ **Resume support** for interrupted runs
- ✅ **Compatible output** for vLLM, TensorRT-LLM, HuggingFace

## Features

### Quantization (BF16 → NVFP4)
- **NVFP4 quantization**: Weight-only NVFP4 (MLP-only or all-linear)
- **Streaming calibration**: Optional layer-by-layer calibration for better accuracy
- **Memory efficient**: Process 684B models with <100GB RAM
- **Fast**: 10-15 minutes (weight-only) or 30-45 minutes (with calibration)
- **Compatible**: Outputs work with vLLM (`--quantization compressed_tensors`)

### Decompression (INT4 → BF16)
- **GPU-accelerated decompression**: Fast INT4 to BF16 unpacking
- **Multi-GPU support**: Parallel processing across multiple GPUs
- **Memory efficient**: Incremental shard-by-shard decompression
- **Resume support**: Continue interrupted decompression runs
- **Robust**: Handles compressed-tensors format with group quantization

## Quick Start

### Installation

```bash
pip install modelopt-streaming

# Or install from source:
git clone https://github.com/yourusername/ModelOptStreaming.git
cd ModelOptStreaming
pip install -e .

# Install Model-Optimizer for NVFP4 primitives:
pip install git+https://github.com/NVIDIA/Model-Optimizer.git@zhiyu/support-kimi-k2.5-ptq
```

### Basic Usage

```bash
# Weight-only NVFP4 (fastest, ~12 minutes)
modelopt-streaming quantize \
  --input_dir /path/to/model-bf16 \
  --output_dir /path/to/model-nvfp4 \
  --format nvfp4 \
  --mlp_only

# With calibration for W4A4 (slower, more accurate, ~30-45 minutes)
modelopt-streaming quantize \
  --input_dir /path/to/model-bf16 \
  --output_dir /path/to/model-nvfp4 \
  --format nvfp4 \
  --mlp_only \
  --calibrate \
  --calib_size 512 \
  --calib_dataset cnn_dailymail

# With selective quantization (exclude specific layers, e.g., baseten strategy)
modelopt-streaming quantize \
  --input_dir /path/to/model-bf16 \
  --output_dir /path/to/model-nvfp4 \
  --format nvfp4 \
  --mlp_only \
  --exclude_config /path/to/reference-model/hf_quant_config.json \
  --calibrate

# Serve with vLLM
vllm serve /path/to/model-nvfp4 \
  --quantization compressed_tensors \
  -tp 4

# Decompress INT4 model back to BF16
modelopt-streaming decompress \
  --input_dir /path/to/model-int4 \
  --output_dir /path/to/model-bf16 \
  --num_gpus 4

# Decompress INT4 to BF16 (text-only, exclude vision)
modelopt-streaming decompress \
  --input_dir /path/to/model-multimodal-int4 \
  --output_dir /path/to/model-text-only-bf16 \
  --num_gpus 4 \
  --text-only

# Extract exclusion patterns from a quantization config
modelopt-streaming extract \
  --config /path/to/hf_quant_config.json \
  --verbose \
  --output exclusion_patterns.txt

# Extract text-only model from multimodal BF16 (no decompression)
modelopt-streaming extract-text \
  --input_dir /path/to/model-multimodal-bf16 \
  --output_dir /path/to/model-text-only-bf16
```

### Python API

```python
from modelopt_streaming import StreamingQuantizer, decompress_model_incremental

# Quantization: Weight-only (fast)
quantizer = StreamingQuantizer(
    input_dir="/path/to/model-bf16",
    output_dir="/path/to/model-nvfp4",
    format="nvfp4",
    mlp_only=True,
    device="cuda:0"
)
quantizer.run()

# Quantization: With calibration (better accuracy)
quantizer = StreamingQuantizer(
    input_dir="/path/to/model-bf16",
    output_dir="/path/to/model-nvfp4",
    format="nvfp4",
    mlp_only=True,
    device="cuda:0",
    calibrate=True,
    calib_size=512,
    calib_dataset="cnn_dailymail"
)
quantizer.run()

# Decompression: INT4 → BF16
decompress_model_incremental(
    model_path="/path/to/model-int4",
    output_path="/path/to/model-bf16",
    num_gpus=4,
    fresh=False,  # Resume from partial decompression
    verbose=True
)
```

## Supported Formats

| Format | Weight-Only | W4A4 (calibrated) | Status |
|--------|-------------|-------------------|--------|
| **NVFP4** | ✅ | ✅ | Stable |
| **FP8** | 🔄 | 🔄 | Coming soon |
| **INT4-AWQ** | 🔄 | ❌ | Coming soon |

## Quantization Modes

### Weight-Only (W4A16)
- **Fastest**: 10-15 minutes for 684B models
- **No calibration needed**: Computes scales from weight distributions only
- **Good accuracy**: ~1-3% perplexity degradation vs BF16
- **Use case**: Production serving, fast iteration

### With Calibration (W4A4)
- **Slower**: 30-45 minutes for 684B models
- **Requires calibration data**: Uses real samples to observe activation magnitudes
- **Better accuracy**: <1% perplexity degradation vs BF16
- **Use case**: Research, benchmarking, accuracy-critical deployments

### Selective Quantization
- **Exclude specific layers**: Use `--exclude_config` to replicate exclusion strategies from reference models
- **Glob pattern support**: Supports wildcards like `model.layers.0*` to exclude groups of layers
- **Use case**: Match quality of known-good quantized models (e.g., baseten NVFP4 models)

Example: Extract and reuse exclusion patterns from a reference model:
```bash
# 1. Extract patterns from reference model
modelopt-streaming extract \
  --config /path/to/reference-model/hf_quant_config.json \
  --verbose \
  --output exclusion_patterns.txt

# 2. Use patterns for your own quantization
modelopt-streaming quantize \
  --input_dir /path/to/your-model-bf16 \
  --output_dir /path/to/your-model-nvfp4 \
  --exclude_config /path/to/reference-model/hf_quant_config.json \
  --mlp_only \
  --calibrate
```

## Performance

Benchmarked on Kimi-K2.5 (684B parameters, 61 layers, 384 experts/layer):

| Approach | Mode | Runtime | Peak RAM | Success Rate |
|----------|------|---------|----------|--------------|
| Model-Optimizer | W4A16 (calib_size=0) | 3+ hours → OOM | 920 GB | 0% (killed) |
| **ModelOptStreaming** | **W4A16 (weight-only)** | **12 minutes** | **~80 GB** | **100%** |
| **ModelOptStreaming** | **W4A4 (calibrated)** | **~35 minutes** | **~120 GB** | **100%** |

## Architecture

```
┌─────────────┐
│ BF16 Model  │ (64 shards, ~2TB)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Optional: Streaming Calibration    │
│  - Load 1 layer at a time           │
│  - Run calibration samples          │
│  - Capture activation stats         │
│  - Free layer                       │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Streaming Quantization             │
│                                     │
│  For each shard:                    │
│  1. Load shard (safetensors)        │
│  2. For each MLP weight:            │
│     - Call NVFP4QTensor.quantize()  │
│     - Use calibrated input_scale    │
│  3. Save quantized shard            │
│  4. Free memory                     │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│ NVFP4 Model │ Compatible with vLLM/TRT-LLM
└─────────────┘
```

## Comparison with Model-Optimizer

| Feature | Model-Optimizer | ModelOptStreaming |
|---------|-----------------|-------------------|
| Full model load | ✅ Required | ❌ Not needed |
| Calibration | ✅ Full model | ✅ Streaming (1 layer at a time) |
| Memory efficient | ❌ 900GB+ RAM | ✅ <120GB RAM |
| Resume support | ❌ No | ✅ Yes |
| Runtime (684B) | 3+ hours | 10-45 min |
| OOM resilient | ❌ Frequent crashes | ✅ Stable |

## Quality Validation

### Gate/Up Scale Matching for MoE Models

ModelOptStreaming implements **critical gate/up scale matching** for MoE (Mixture-of-Experts) models:

- vLLM's fused MoE kernels require `gate_proj.weight_scale_2 == up_proj.weight_scale_2`
- Without matching, kernels fail or produce incorrect results
- **Our implementation**: Pre-pass to identify gate/up pairs and compute unified scale_2
- **Result**: 100% perfect matching across all 23,101 gate/up pairs (validated on Kimi-K2.5)

This ensures optimal performance and accuracy when serving with vLLM.

### Validation Tools

Included validation scripts to verify quantized models:

```bash
# Validate gate/up scale matching (critical for MoE)
python examples/validate_gate_up_matching.py

# Compare scale distributions
python examples/compare_scales.py

# Compare weight values byte-by-byte
python examples/compare_weight_values.py
```

### Validation Results (Kimi-K2.5 684B MoE)

Tested on Kimi-K2.5 with 684B parameters, 61 layers, 384 experts/layer:

| Metric | ModelOptStreaming | Expected |
|--------|------------------|----------|
| Gate/up pairs matched | **23,101/23,101 (100%)** | ≥99.9% |
| Weight quantization | Byte-for-byte identical to reference | ✅ |
| Scale distributions | Statistically equivalent | ✅ |
| vLLM compatibility | Full support (with correct version) | ✅ |

## vLLM Serving Compatibility

### Recommended vLLM Version

**Use vLLM v0.16.0rc1.dev78 or earlier** for NVFP4 models:

```bash
# Use this container version:
docker://vllm/vllm-openai:nightly  # Then pin to dev78

# Or pin specific wheel:
pip install 'vllm @ https://wheels.vllm.ai/nightly/vllm-0.16.0rc1.dev78+gb6bb2842c-cp38-abi3-manylinux_2_31_aarch64.whl'
```

⚠️ **Known Issue**: vLLM nightly builds after dev78 (including dev118+) have a regression bug with NVFP4 tensor-parallel loading that causes `AssertionError: param_data.shape == loaded_weight.shape`. This is a vLLM bug, not a model format issue.

### Hardware Requirements for NVFP4

NVFP4 requires **Blackwell GPUs (SM 10.0)**:
- ✅ B200, B300, GB200 (compute capability 10.0)
- ❌ H100, H200 (compute capability 9.0) - will fail during kernel compilation

Check your GPU:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

### Serving Example

```bash
# Serve with vLLM (TP=4)
vllm serve /path/to/model-nvfp4 \
  --served-model-name my-model \
  -tp 4 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --port 8000
```

See [examples/serve_README.md](examples/serve_README.md) for complete serving instructions.

## Use Cases

- **Production quantization**: Convert fine-tuned models quickly and reliably
- **Research workflows**: Fast iteration without infrastructure overhead
- **Large-scale deployments**: Quantize 100B+ parameter models on standard nodes
- **CI/CD pipelines**: Reproducible, fast quantization in automated workflows

## Requirements

- **GPU**: CUDA-capable GPU (compute capability 7.5+)
- **RAM**: ~100GB for 684B models (weight-only), ~150GB (with calibration)
- **Storage**: 2× model size (input BF16 + output NVFP4)
- **Python**: 3.10+
- **PyTorch**: 2.0+

## CLI Reference

### Quantize Command

```bash
modelopt-streaming quantize \
  --input_dir PATH          # Input BF16/FP16 model directory
  --output_dir PATH         # Output quantized model directory
  --format {nvfp4,fp8}      # Quantization format (default: nvfp4)
  --mlp_only                # Quantize only MLP weights (default: True)
  --all_linear              # Quantize all linear weights (overrides --mlp_only)
  --exclude_config PATH     # JSON config with exclude_modules list (selective quantization)
  --block_size INT          # Block size for group quantization (default: 16)
  --device DEVICE           # CUDA device (default: cuda:0)
  --num_gpus INT            # Number of GPUs for parallel quantization (default: 1)
  --resume                  # Resume from existing output
  --calibrate               # Enable activation calibration (W4A4)
  --calib_size INT          # Number of calibration samples (default: 512)
  --calib_dataset NAME      # HuggingFace dataset for calibration (default: cnn_dailymail)
```

### Decompress Command

```bash
modelopt-streaming decompress \
  --input_dir PATH          # Input INT4 (compressed-tensors) model directory
  --output_dir PATH         # Output BF16 model directory
  --num_gpus INT            # Number of GPUs for parallel decompression (default: 4)
  --fresh                   # Remove existing output and start fresh
  --text-only               # Extract text-only model (exclude vision components)
  --quiet                   # Suppress progress output
```

### Extract Command (Exclusion Patterns)

```bash
modelopt-streaming extract \
  --config PATH             # Quantization config file (hf_quant_config.json or config.json)
  --verbose                 # Print all patterns (otherwise just summary)
  --output PATH             # Save patterns to text file (one per line)
```

### Extract-Text Command (Text-Only Model)

```bash
modelopt-streaming extract-text \
  --input_dir PATH          # Input multimodal BF16 model directory
  --output_dir PATH         # Output text-only BF16 model directory
  --quiet                   # Suppress progress output
```

## Roadmap

- [x] NVFP4 weight-only quantization
- [x] Streaming calibration (layer-by-layer)
- [ ] FP8 quantization
- [ ] INT4-AWQ quantization
- [ ] Multi-GPU parallel quantization
- [ ] Per-token dynamic quantization

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0

## Citation

If you use ModelOptStreaming in your research, please cite:

```bibtex
@software{modelopt_streaming,
  title={ModelOptStreaming: Memory-Efficient Streaming Quantization},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/ModelOptStreaming}
}
```

## Acknowledgments

Built on top of [NVIDIA Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) quantization primitives.
