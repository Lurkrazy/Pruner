# TVM Vendor Benchmark Docker Guide

This guide explains how to use Docker to reproduce the vendor library benchmark comparisons mentioned in the Pruner paper.

## Quick Start

### Build the Docker Image

```bash
cd /path/to/tvm
docker build -f docker/Dockerfile.vendor_benchmark -t tvm-vendor-benchmark .
```

### Run the Container

For GPU benchmarks (recommended):
```bash
docker run --gpus all -it --rm tvm-vendor-benchmark
```

For CPU-only benchmarks:
```bash
docker run -it --rm tvm-vendor-benchmark
```

### Alternative: Docker Compose

For easier management, use the provided docker-compose file:

```bash
# GPU version
docker-compose -f docker/docker-compose.vendor-benchmark.yml up vendor-benchmark

# CPU version  
docker-compose -f docker/docker-compose.vendor-benchmark.yml up vendor-benchmark-cpu
```

### Quick Start Script

Use the convenience script for a streamlined experience:

```bash
# Build the image
./docker/run_vendor_benchmark.sh build

# Run the container
./docker/run_vendor_benchmark.sh run
```

## Available Commands in Container

Once inside the container, you have access to several convenience commands:

### Check Available Libraries
```bash
test-vendors
```
This will show which vendor libraries are installed and working.

### Run Individual Benchmarks
```bash
# PyTorch benchmark
run-benchmark pytorch_benchmark.py --network resnet-50 --repeat 100

# ONNX Runtime benchmark  
run-benchmark onnxruntime_benchmark.py --network resnet-50 --repeat 100

# TVM vs PyTorch comparison
run-benchmark tvm_vs_pytorch.py --network resnet-50 --repeat 100
```

### Run Comprehensive Comparison
```bash
# Compare multiple vendors
run-benchmark vendor_comparison_suite.py \
    --vendors pytorch,onnxruntime \
    --repeat 100 \
    --output results.json
```

## TensorRT Setup

TensorRT requires manual installation due to NVIDIA's licensing requirements:

1. Run the setup helper:
```bash
tensorrt-setup
```

2. Download TensorRT from [NVIDIA Developer Portal](https://developer.nvidia.com/tensorrt)

3. Mount the TensorRT archive into the container:
```bash
docker run --gpus all -it --rm \
    -v /path/to/TensorRT-x.x.x.tar.gz:/tmp/tensorrt.tar.gz \
    tvm-vendor-benchmark
```

4. Inside the container, extract and install:
```bash
cd /tmp
tar -xzf tensorrt.tar.gz
export TENSORRT_ROOT=/tmp/TensorRT-x.x.x
pip install $TENSORRT_ROOT/python/tensorrt-*-py3-none-any.whl
```

5. Verify TensorRT installation:
```bash
test-vendors
```

## Reproducing Paper Results

### Basic Performance Comparison
```bash
# Get PyTorch baseline (as mentioned in paper)
run-benchmark pytorch_benchmark.py --network resnet-50 --device cuda --repeat 600

# Get TVM performance  
cd /workspace/apps/benchmark
python gpu_imagenet_bench.py --network resnet-50 --target "cuda" --repeat 600

# Direct comparison
cd vendor_comparison
run-benchmark tvm_vs_pytorch.py --network resnet-50 --repeat 600
```

### Comprehensive Vendor Comparison
```bash
# Compare all available vendors
run-benchmark vendor_comparison_suite.py \
    --vendors pytorch,onnxruntime,tensorrt \
    --networks resnet-50,resnet-18,mobilenet \
    --repeat 600 \
    --output paper_reproduction_results.json
```

### Export Results
```bash
# Copy results from container to host
docker cp container_id:/workspace/apps/benchmark/vendor_comparison/paper_reproduction_results.json ./
```

## Volume Mounting for Data Persistence

To persist results and data between container runs:

```bash
docker run --gpus all -it --rm \
    -v $(pwd)/benchmark_results:/workspace/results \
    tvm-vendor-benchmark
```

Then save results to `/workspace/results/` inside the container.

## Development and Customization

To modify benchmark scripts or add new vendor libraries:

```bash
# Mount source code for development
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    tvm-vendor-benchmark
```

This allows you to edit the benchmark scripts on your host machine and run them immediately in the container.

## Hardware Requirements

- **GPU Benchmarks**: NVIDIA GPU with CUDA 11.0+ support
- **Memory**: At least 8GB RAM recommended for large models
- **Storage**: 20GB+ for Docker image and model caches
- **Docker**: Version 19.03+ with nvidia-docker2 support

## Troubleshooting

### CUDA Issues
If you encounter CUDA-related errors:
```bash
# Check NVIDIA Docker setup
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Memory Issues
For large models, increase Docker memory limits:
```bash
# Run with memory limit
docker run --gpus all -it --rm --memory=16g tvm-vendor-benchmark
```

### Permission Issues
If you encounter permission issues with mounted volumes:
```bash
# Run with current user
docker run --gpus all -it --rm --user $(id -u):$(id -g) \
    -v $(pwd):/workspace tvm-vendor-benchmark
```

## Container Specifications

- **Base Image**: nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
- **Python**: 3.6+
- **TVM**: Built from source with CUDA support
- **PyTorch**: 1.8.1 with CUDA 11.1 support
- **ONNX Runtime**: GPU-enabled version
- **Size**: Approximately 8-10GB

## Contributing

To add support for new vendor libraries:

1. Modify `docker/Dockerfile.vendor_benchmark`
2. Add installation commands for the new library
3. Update the dependency test in `test_dependencies.py`
4. Create benchmark script following existing patterns
5. Update this documentation

## License

This Docker setup follows the same Apache 2.0 license as the main TVM project.