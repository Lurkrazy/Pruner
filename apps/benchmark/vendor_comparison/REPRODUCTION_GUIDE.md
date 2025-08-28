# Reproducing Vendor Library Benchmarks for Pruner Paper

This directory provides the benchmark infrastructure needed to reproduce the vendor library performance comparisons mentioned in the Pruner paper. The paper references benchmarks against TensorRT, PyTorch, and other vendor libraries, and these scripts allow you to run those same comparisons on your hardware.

## Quick Start with Docker (Recommended)

The easiest way to reproduce all benchmark results is using Docker:

```bash
# Build and run the container
./docker/run_vendor_benchmark.sh build
./docker/run_vendor_benchmark.sh run

# Inside container, check available libraries
test-vendors

# Run the paper's main comparison
run-benchmark tvm_vs_pytorch.py --network resnet-50 --repeat 600
```

📋 **See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for complete Docker setup instructions.**

## Manual Setup (Alternative)

### Check Dependencies
```bash
cd apps/benchmark/vendor_comparison
python test_dependencies.py
```

### Run Basic Comparison
```bash
# Compare TVM vs PyTorch (most commonly available)
python tvm_vs_pytorch.py --network resnet-50 --repeat 100
```

### Run Comprehensive Comparison
```bash
# Compare TVM against all available vendor libraries
python vendor_comparison_suite.py --network resnet-50 --repeat 100 --vendors pytorch,onnxruntime
```

## Paper Reproduction Workflow

### Step 1: Environment Setup

The paper likely used a setup similar to this:

```bash
# Install TVM with CUDA support (for GPU benchmarks)
# See main TVM documentation for installation

# Install vendor libraries for comparison
pip install torch torchvision              # PyTorch
pip install onnxruntime-gpu onnx            # ONNX Runtime  
# TensorRT requires manual installation from NVIDIA

# Verify installation
python test_dependencies.py
```

### Step 2: Reproduce Performance Baselines

To reproduce the vendor library baselines mentioned in the paper:

```bash
# Get PyTorch native performance (baseline)
python pytorch_benchmark.py --network resnet-50 --device cuda --repeat 600

# Get ONNX Runtime native performance (baseline)  
python onnxruntime_benchmark.py --network resnet-50 --provider cuda --repeat 600

# Get TensorRT native performance (baseline, if available)
python tensorrt_benchmark.py --network resnet-50 --precision fp32 --repeat 600
```

### Step 3: Get TVM Performance with Pruner Optimizations

The main TVM benchmarks with Pruner can be run using the existing infrastructure:

```bash
# From the main directory, run TVM with Pruner optimizations
python tune_network.py --network resnet_50 --n-trials 2000 --cost-model pam --target "cuda --model=a100" --psa a100_40

# Then benchmark the optimized model
cd apps/benchmark
python gpu_imagenet_bench.py --network resnet-50 --model 1080ti --repeat 600
```

### Step 4: Direct Comparison

Compare TVM (with Pruner) against vendor libraries:

```bash
cd apps/benchmark/vendor_comparison

# Compare specific frameworks
python vendor_comparison_suite.py \
    --network resnet-50 \
    --device cuda \
    --repeat 600 \
    --vendors pytorch,onnxruntime,tensorrt \
    --output results_resnet50.json
```

### Step 5: Reproduce Paper Networks

Run the same networks mentioned in the paper:

```bash
# Classification networks from paper
for network in resnet-50 inception_v3 mobilenet densenet-121; do
    python vendor_comparison_suite.py \
        --network $network \
        --device cuda \
        --repeat 600 \
        --vendors pytorch,onnxruntime \
        --output results_${network}.json
done
```

## Understanding the Results

### Output Format

The benchmark scripts output results in a consistent format:

```
Network              TVM (ms)        PyTorch (ms)    Speedup   
--------------------------------------------------
resnet-50           12.34           15.67           1.27x
mobilenet           5.21            7.89            1.51x
```

### Performance Metrics

- **Mean Time (ms)**: Average inference time in milliseconds
- **Std Dev (ms)**: Standard deviation of inference times
- **Speedup**: Ratio of vendor time to TVM time (>1 means TVM is faster)

### Expected Results

Based on typical TVM performance:
- TVM should show speedups of 1.2-2x over PyTorch on many models
- Results will vary based on hardware, model architecture, and optimization level
- GPU results typically show larger speedups than CPU results

## Hardware-Specific Reproduction

### NVIDIA A100 (Paper Hardware)

```bash
# Use A100-specific model settings
python vendor_comparison_suite.py \
    --network resnet-50 \
    --device cuda \
    --repeat 600 \
    --vendors pytorch,onnxruntime,tensorrt \
    --tensorrt-precision fp16
```

### Other NVIDIA GPUs

```bash
# Adjust model parameter for your GPU
cd apps/benchmark
python gpu_imagenet_bench.py --model 1080ti  # or titanx, tx2, etc.
```

### CPU Benchmarks

```bash
# CPU-only comparison
python vendor_comparison_suite.py \
    --network resnet-50 \
    --device cpu \
    --vendors pytorch,onnxruntime
```

## Troubleshooting

### Common Issues

1. **"CUDA not available"**: Install CUDA toolkit and ensure GPU drivers are current
2. **"TensorRT not found"**: TensorRT requires manual installation from NVIDIA Developer Portal
3. **Memory errors**: Reduce batch size or model size for GPU memory limits
4. **Slow performance**: Ensure you're using optimized builds (not debug builds)

### Dependency Issues

If libraries are missing, install them incrementally:

```bash
# Minimum for PyTorch comparison
pip install torch torchvision

# Add ONNX Runtime
pip install onnxruntime-gpu onnx

# TensorRT (manual installation required)
# Download from: https://developer.nvidia.com/tensorrt
```

## Paper Citation Context

When using these benchmarks to reproduce or extend the Pruner paper results, ensure you:

1. **Use similar hardware**: The paper used NVIDIA A100 GPUs primarily
2. **Use consistent settings**: Match batch sizes, precision, and optimization levels
3. **Report variance**: Include standard deviations in performance comparisons
4. **Document environment**: Record software versions and hardware specifications

## Advanced Usage

### Custom Networks

To benchmark your own models:

1. Add the model to the `get_network()` function in `../util.py`
2. Implement PyTorch version in vendor benchmark scripts
3. Run comparison suite with your custom network

### Precision Comparisons

```bash
# Compare different precisions with TensorRT
python tensorrt_benchmark.py --network resnet-50 --precision fp16
python tensorrt_benchmark.py --network resnet-50 --precision int8
```

### Batch Size Studies

Modify the input shapes in benchmark scripts to test different batch sizes:

```python
# In benchmark scripts, change:
input_shape = (batch_size, 3, 224, 224)  # Instead of (1, 3, 224, 224)
```

This infrastructure should provide everything needed to reproduce and extend the vendor library comparisons mentioned in the Pruner paper.