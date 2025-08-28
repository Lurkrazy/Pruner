# Vendor Library Benchmarks - Summary

## Overview

This directory contains the missing vendor library benchmark infrastructure that was referenced in the Pruner paper but not previously included in the repository. The paper mentions performance comparisons against TensorRT, PyTorch, and other vendor libraries, but the original repository only contained TVM benchmark code.

## What Was Missing

The original repository had:
- ✅ TVM benchmark infrastructure (`apps/benchmark/gpu_imagenet_bench.py`)
- ✅ Vendor library integration code (TensorRT, PyTorch, ONNX frontends)
- ✅ Correctness tests for vendor libraries
- ❌ **Performance benchmarks for native vendor libraries**
- ❌ **Side-by-side comparison scripts**
- ❌ **Paper reproduction guidance**

## What This Adds

This benchmark suite provides:

### 1. Native Vendor Library Benchmarks
- `pytorch_benchmark.py` - Benchmark native PyTorch models
- `onnxruntime_benchmark.py` - Benchmark native ONNX Runtime models  
- `tensorrt_benchmark.py` - Benchmark native TensorRT engines

### 2. Comparison Scripts
- `vendor_comparison_suite.py` - Compare TVM against multiple vendor libraries
- `tvm_vs_pytorch.py` - Direct TVM vs PyTorch comparison

### 3. Testing and Documentation
- `test_dependencies.py` - Check which vendor libraries are available
- `README.md` - Usage guide for all scripts
- `REPRODUCTION_GUIDE.md` - Detailed paper reproduction instructions

## Key Benefits

1. **Reproduces Paper Claims**: Allows researchers to verify vendor library performance claims
2. **Fair Comparisons**: Uses identical models and data across all frameworks
3. **Research Reusability**: Provides infrastructure for extending comparisons to new models
4. **Hardware Flexibility**: Works on both CUDA and CPU backends
5. **Easy Integration**: Follows existing TVM benchmark patterns

## Supported Networks

All scripts support the same networks as the existing TVM benchmarks:
- ResNet-18, ResNet-34, ResNet-50
- VGG-16, VGG-19
- DenseNet-121
- Inception v3
- MobileNet v2
- SqueezeNet v1.0, v1.1

## Example Usage

```bash
# Check what's available
python test_dependencies.py

# Quick TVM vs PyTorch comparison
python tvm_vs_pytorch.py --network resnet-50

# Comprehensive comparison with results export
python vendor_comparison_suite.py \
    --network resnet-50 \
    --vendors pytorch,onnxruntime \
    --repeat 600 \
    --output comparison_results.json
```

## Expected Performance Patterns

Based on typical TVM performance characteristics:
- **TVM vs PyTorch**: 1.2-2x speedup expected on most models
- **TVM vs ONNX Runtime**: 1.1-1.5x speedup expected  
- **TVM vs TensorRT**: Competitive, varies by model and precision
- **GPU vs CPU**: Larger speedups typically observed on GPU

## Research Applications

This infrastructure enables:

1. **Paper Validation**: Reproduce performance claims from the Pruner paper
2. **Baseline Establishment**: Get vendor library performance baselines for new research
3. **Hardware Studies**: Compare framework performance across different GPU architectures
4. **Model Analysis**: Understand which models benefit most from TVM optimizations
5. **Precision Studies**: Compare different numerical precisions (fp32, fp16, int8)

## Integration with Existing TVM Benchmarks

These scripts complement the existing TVM benchmark infrastructure:

- **Existing**: `apps/benchmark/gpu_imagenet_bench.py` (TVM performance)
- **New**: `apps/benchmark/vendor_comparison/` (vendor baselines + comparisons)
- **Together**: Complete performance evaluation ecosystem

## Future Extensions

The framework is designed to be easily extensible:
- Add new vendor libraries (e.g., QDN when available)
- Add new model architectures
- Add new metrics beyond inference time
- Add batch size studies
- Add memory usage comparisons

This infrastructure provides researchers with the tools needed to conduct comprehensive, fair comparisons between TVM (with Pruner optimizations) and vendor libraries, supporting both paper reproduction and new research directions.