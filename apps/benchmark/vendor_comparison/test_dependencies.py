#!/usr/bin/env python3
"""Test script to verify vendor benchmark dependencies and basic functionality."""

import sys
import os

def test_dependencies():
    """Test which vendor libraries are available."""
    print("Testing vendor library dependencies...")
    print("=" * 50)
    
    # Test TVM
    try:
        import tvm
        print(f"✓ TVM: {tvm.__version__}")
        tvm_available = True
    except ImportError as e:
        print(f"✗ TVM: Not available ({e})")
        tvm_available = False
    
    # Test PyTorch
    try:
        import torch
        import torchvision
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ TorchVision: {torchvision.__version__}")
        pytorch_available = True
    except ImportError as e:
        print(f"✗ PyTorch: Not available ({e})")
        pytorch_available = False
    
    # Test ONNX Runtime
    try:
        import onnxruntime as ort
        import onnx
        print(f"✓ ONNX Runtime: {ort.__version__}")
        print(f"✓ ONNX: {onnx.__version__}")
        onnxruntime_available = True
    except ImportError as e:
        print(f"✗ ONNX Runtime: Not available ({e})")
        onnxruntime_available = False
    
    # Test TensorRT
    try:
        import tensorrt as trt
        import pycuda
        print(f"✓ TensorRT: {trt.__version__}")
        print(f"✓ PyCUDA: {pycuda.VERSION_TEXT}")
        tensorrt_available = True
    except ImportError as e:
        print(f"✗ TensorRT: Not available ({e})")
        tensorrt_available = False
    
    print("=" * 50)
    
    # Test hardware
    print("Testing hardware availability...")
    print("-" * 30)
    
    if tvm_available:
        cuda_available = tvm.runtime.enabled("cuda") and tvm.gpu(0).exist
        print(f"CUDA (TVM): {'✓' if cuda_available else '✗'}")
    
    if pytorch_available:
        torch_cuda = torch.cuda.is_available()
        print(f"CUDA (PyTorch): {'✓' if torch_cuda else '✗'}")
        if torch_cuda:
            print(f"  GPU: {torch.cuda.get_device_name()}")
    
    if onnxruntime_available:
        import onnxruntime as ort
        cuda_provider = 'CUDAExecutionProvider' in ort.get_available_providers()
        print(f"CUDA (ONNX Runtime): {'✓' if cuda_provider else '✗'}")
    
    print("=" * 50)
    
    return {
        'tvm': tvm_available,
        'pytorch': pytorch_available,
        'onnxruntime': onnxruntime_available,
        'tensorrt': tensorrt_available
    }

def run_quick_test():
    """Run a quick functionality test with available frameworks."""
    available = test_dependencies()
    
    print("\nRunning quick functionality tests...")
    print("-" * 40)
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if available['pytorch']:
        try:
            print("Testing PyTorch benchmark...")
            from pytorch_benchmark import benchmark_pytorch
            
            # Quick test with small number of iterations
            mean_time, std_time = benchmark_pytorch('mobilenet', 'cpu', repeat=3, warmup=1)
            print(f"  PyTorch MobileNet (CPU): {mean_time:.2f} ± {std_time:.2f} ms")
            
        except Exception as e:
            print(f"  PyTorch test failed: {e}")
    
    if available['tvm']:
        try:
            print("Testing TVM benchmark...")
            from vendor_comparison_suite import run_framework_benchmark
            
            result = run_framework_benchmark('tvm', 'mobilenet', 'cpu', repeat=3, warmup=1)
            if result:
                mean_time, std_time = result
                print(f"  TVM MobileNet (CPU): {mean_time:.2f} ± {std_time:.2f} ms")
            else:
                print("  TVM test failed")
                
        except Exception as e:
            print(f"  TVM test failed: {e}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    run_quick_test()