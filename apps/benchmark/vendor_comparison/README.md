# Vendor Library Benchmark Comparison

This directory contains benchmark scripts for comparing TVM performance against vendor libraries mentioned in the Pruner paper, including TensorRT, PyTorch, and ONNX Runtime.

## Available Benchmarks

### TensorRT Comparison
- `tensorrt_benchmark.py` - Native TensorRT performance benchmarking
- `tvm_vs_tensorrt.py` - Side-by-side comparison of TVM and TensorRT

### PyTorch Comparison  
- `pytorch_benchmark.py` - Native PyTorch performance benchmarking
- `tvm_vs_pytorch.py` - Side-by-side comparison of TVM and PyTorch

### ONNX Runtime Comparison
- `onnxruntime_benchmark.py` - Native ONNX Runtime performance benchmarking
- `tvm_vs_onnxruntime.py` - Side-by-side comparison of TVM and ONNX Runtime

### Unified Comparison
- `vendor_comparison_suite.py` - Comprehensive benchmark suite comparing TVM against all vendor libraries

## Requirements

To run these benchmarks, you need to install the corresponding vendor libraries:

```bash
# TensorRT (requires NVIDIA GPU and CUDA)
pip install pycuda
# Download and install TensorRT from NVIDIA Developer Portal

# PyTorch
pip install torch torchvision

# ONNX Runtime  
pip install onnxruntime-gpu  # for GPU
# or
pip install onnxruntime      # for CPU
```

## Usage

### Run Individual Vendor Benchmarks
```bash
# Benchmark TensorRT native performance
python tensorrt_benchmark.py --network resnet-50 --repeat 100

# Benchmark PyTorch native performance  
python pytorch_benchmark.py --network resnet-50 --repeat 100

# Benchmark ONNX Runtime native performance
python onnxruntime_benchmark.py --network resnet-50 --repeat 100
```

### Run TVM vs Vendor Comparisons
```bash
# Compare TVM vs TensorRT
python tvm_vs_tensorrt.py --network resnet-50 --repeat 100

# Compare TVM vs PyTorch
python tvm_vs_pytorch.py --network resnet-50 --repeat 100

# Compare TVM vs ONNX Runtime
python tvm_vs_onnxruntime.py --network resnet-50 --repeat 100
```

### Run Comprehensive Comparison Suite
```bash
# Compare TVM against all vendor libraries
python vendor_comparison_suite.py --network resnet-50 --repeat 100 --vendors tensorrt,pytorch,onnxruntime
```

## Supported Networks

- resnet-18, resnet-34, resnet-50
- vgg-16, vgg-19  
- densenet-121
- inception_v3
- mobilenet
- squeezenet_v1.0, squeezenet_v1.1

## Output Format

All benchmark scripts output results in a consistent format:

```
--------------------------------------------------
Framework       Network         Mean Time (std)   
--------------------------------------------------
TVM            resnet-50        12.34 ms (1.23 ms)
TensorRT       resnet-50        10.45 ms (0.98 ms)
PyTorch        resnet-50        15.67 ms (2.01 ms)
ONNX Runtime   resnet-50        11.23 ms (1.45 ms)
--------------------------------------------------
```

## Notes

- Ensure you have the appropriate hardware (NVIDIA GPU for TensorRT/CUDA benchmarks)
- Results may vary based on hardware, drivers, and library versions
- The benchmarks focus on inference performance with batch size 1
- All frameworks use the same input data and model architectures for fair comparison