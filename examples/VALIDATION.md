# Model Validation Guide

This directory contains validation scripts to verify the quality and correctness of quantized NVFP4 models.

## Validation Scripts

### 1. Gate/Up Scale Matching (`validate_gate_up_matching.py`)

**Purpose**: Verify that gate_proj and up_proj in MoE models have matching weight_scale_2 values.

**Why it matters**: vLLM's fused MoE kernels require `gate_proj.weight_scale_2 == up_proj.weight_scale_2`. Mismatches cause kernel failures or degraded accuracy.

**Usage**:
```bash
python validate_gate_up_matching.py

# Or edit the script to compare custom models:
# MODEL_A = "/path/to/first-model"
# MODEL_B = "/path/to/second-model"
```

**Expected Output**:
```
✅ Matched pairs: 23101/23101
❌ Mismatched pairs: 0/23101

✅ ALL GATE/UP PAIRS PERFECTLY MATCHED!
```

**Interpretation**:
- ✅ **100% matched**: Perfect, ready for vLLM serving
- ⚠️ **>99% matched**: Acceptable, minor warnings in vLLM
- ❌ **<99% matched**: May cause accuracy issues or kernel failures

### 2. Scale Distribution Analysis (`compare_scales.py`)

**Purpose**: Analyze and compare scale distributions between models.

**What it checks**:
- `weight_scale_2` (global, per-weight scalar)
- `weight_scale` (per-block, FP8 or FP32)
- `input_scale` (activation, scalar)
- Gate/up pair matching in first 3 shards

**Usage**:
```bash
python compare_scales.py

# Edit script to compare specific models
```

**Expected Output**:
```
weight_scale_2 (global, float32 scalar):
  Count: 1901
  Min:   3.578550e-05
  Max:   4.795619e-04
  Mean:  5.839532e-05
  Std:   2.606264e-05
```

**Interpretation**:
- Scales should be in range `[1e-5, 1e-3]` for typical models
- Higher variance indicates diverse weight magnitudes
- Statistics should be similar between models quantized from same source

### 3. Weight Value Comparison (`compare_weight_values.py`)

**Purpose**: Byte-level comparison of quantized weight tensors between two models.

**What it checks**:
- Weight tensor shapes and dtypes
- First 20 packed bytes (should be identical for same source)
- Scale value matching (weight_scale, weight_scale_2)
- Activation scale comparison

**Usage**:
```bash
python compare_weight_values.py

# Edit script to test specific layers
```

**Expected Output**:
```
📊 Weight Tensor:
  NVIDIA:    shape=torch.Size([18432, 3584]), dtype=torch.uint8
  Streaming: shape=torch.Size([18432, 3584]), dtype=torch.uint8
  Shape match: True

  First 20 packed values:
    NVIDIA:    [88, 145, 183, 82, ...]
    Streaming: [88, 145, 183, 82, ...]
    Identical: True
```

**Interpretation**:
- ✅ **Identical packed values**: Same quantization quality
- ✅ **Exact scale match**: Consistent quantization parameters
- ⚠️ **input_scale differs**: Expected for weight-only vs calibrated models

## Validation Workflow

### For New Models

1. **Quantize your model**:
   ```bash
   modelopt-streaming quantize \
     --input_dir /path/to/model-bf16 \
     --output_dir /path/to/model-nvfp4 \
     --mlp_only
   ```

2. **Validate gate/up matching** (MoE models only):
   ```bash
   python examples/validate_gate_up_matching.py
   # Expected: 100% matched pairs
   ```

3. **Inspect scale distributions**:
   ```bash
   python examples/compare_scales.py
   # Check for reasonable scale ranges
   ```

4. **Test serving**:
   ```bash
   # Use vLLM v0.16.0rc1.dev78 or earlier (not latest nightly)
   vllm serve /path/to/model-nvfp4 -tp 4 --trust-remote-code
   ```

### For Debugging Issues

If vLLM serving fails:

1. **Check vLLM version**: Latest nightly has regression bugs
   ```bash
   python -c "import vllm; print(vllm.__version__)"
   # Use: v0.16.0rc1.dev78 (working)
   # Avoid: v0.16.0rc1.dev118+ (broken)
   ```

2. **Validate hardware**: NVFP4 requires SM 10.0 (Blackwell GPUs)
   ```bash
   nvidia-smi --query-gpu=name,compute_cap --format=csv
   # Required: compute_cap = 10.0 (B200/B300/GB200)
   ```

3. **Check gate/up matching**: Run validation script
   ```bash
   python examples/validate_gate_up_matching.py
   # Should show 100% matched pairs
   ```

4. **Inspect logs**: Look for specific errors
   - `AssertionError: param_data.shape == loaded_weight.shape` → vLLM bug (use older version)
   - `w1_weight_scale_2 must match w3_weight_scale_2` → Gate/up mismatch (shouldn't happen with our code)
   - `fatal error: cublasLt.h` → Container missing CUDA dev headers (use nightly, not 0.15)

## Tested Configurations

| Model | Parameters | Experts | Gate/Up Pairs | Match Rate | Status |
|-------|-----------|---------|---------------|------------|--------|
| Kimi-K2.5 | 684B | 23,101 | 23,101 | 100% | ✅ Validated |
| Deepseek-V3 | 671B | ~22,000 | TBD | TBD | 🔄 In progress |

## Known Issues

### vLLM Regression (v0.16.0rc1.dev118+)

**Symptom**: `AssertionError: param_data.shape == loaded_weight.shape` when loading NVFP4 models with TP > 1

**Root Cause**: Bug in vLLM's tensor-parallel weight loading introduced after dev78

**Workaround**: Use vLLM v0.16.0rc1.dev78 or earlier
```bash
pip install 'vllm @ https://wheels.vllm.ai/nightly/vllm-0.16.0rc1.dev78+gb6bb2842c-cp38-abi3-manylinux_2_31_aarch64.whl'
```

**Status**: Reported to vLLM team, fix expected in future nightly builds

## Contact

For issues or questions, please open an issue on GitHub.
