# Serving NVFP4 Models with vLLM

## Setup (one-time)

### Option 1: Interactive Node

```bash
# Allocate GPU node
salloc -N 1 --gres=gpu:4 --mem=0 --time=8:00:00 -A general_cs_infra -p batch

# SSH to node
ssh $(squeue -u $USER -h -o %N | head -n1)

# Setup vLLM
cd /lustre/fsw/portfolios/general/users/asteiner
srun --container-image=nvcr.io/nvidia/pytorch:25.01-py3 \
     --container-mounts=$(pwd):$(pwd),/home/asteiner:/home/asteiner \
     bash ModelOptStreaming/examples/setup_vllm.sh
```

### Option 2: SLURM Batch Job

```bash
cd /lustre/fsw/portfolios/general/projects/general_cs_infra/users/asteiner

export MODEL_PATH=/lustre/fsw/portfolios/general/users/asteiner/Kimi-K2.5-NVFP4-STREAM
export SERVED_MODEL_NAME=kimi-k2.5-nvfp4
export SERVER_PORT=8000

sbatch /lustre/fsw/portfolios/general/users/asteiner/ModelOptStreaming/examples/serve_nvfp4.sbatch
```

## Serve the Model

### Interactive

```bash
# Inside container with vLLM installed:
export MODEL_PATH=/path/to/model-nvfp4
export SERVED_MODEL_NAME=kimi-k2.5-nvfp4
export SERVER_PORT=8000

bash serve_nvfp4.sh
```

### Background (detached)

```bash
nohup bash serve_nvfp4.sh > vllm_server.log 2>&1 &
echo $! > vllm_server.pid

# Monitor
tail -f vllm_server.log

# Stop
kill $(cat vllm_server.pid)
```

## Test the Server

```bash
# Wait for server to be ready (check log for "Application startup complete")
sleep 30

# Run tests
python test_serving.py
```

## Performance Tuning

### Memory Optimization

```bash
# Reduce GPU memory utilization if OOM
vllm serve $MODEL_PATH ... --gpu-memory-utilization 0.85

# Disable multimodal cache if workload is mostly unique images
vllm serve $MODEL_PATH ... --mm-processor-cache-gb 0
```

### Throughput Optimization

```bash
# Enable expert parallelism for MoE models
vllm serve $MODEL_PATH ... --enable-expert-parallel

# Adjust max parallel requests
vllm serve $MODEL_PATH ... --max-num-seqs 256

# Use shared memory for multimodal cache (experimental)
vllm serve $MODEL_PATH ... --mm-processor-cache-type shm
```

### MoE Kernel Tuning

```bash
# Run Triton kernel tuning for your hardware (one-time)
python -m vllm.benchmarks.kernels.benchmark_moe
```

## Monitoring

```bash
# Check server health
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/v1/models

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Troubleshooting

### Server won't start
- Check GPU availability: `nvidia-smi`
- Check port availability: `netstat -tuln | grep $SERVER_PORT`
- Increase timeout in client: `timeout=3600`

### OOM during serving
- Reduce `--gpu-memory-utilization` (default 0.9)
- Reduce `--max-num-seqs`
- Increase `--swap-space` for CPU offload

### Slow multimodal inference
- Use `--mm-encoder-tp-mode data` (vision encoder data parallel)
- Disable cache: `--mm-processor-cache-gb 0` for unique-image workloads
- Enable cache: `--mm-processor-cache-type shm` for repeated images

## API Usage

See [test_serving.py](test_serving.py) for complete examples:
- Text completion
- Multimodal (image + text)
- Streaming responses
