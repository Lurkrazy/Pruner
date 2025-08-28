#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Benchmark script for ONNX Runtime model inference.

This script benchmarks ONNX Runtime models directly,
providing baseline performance numbers for comparison with TVM.
"""
import argparse
import time
import numpy as np
import sys
import os
import tempfile

try:
    import onnxruntime as ort
    import onnx
    import torch
    import torchvision.models as models
except ImportError as e:
    print(f"Required library not installed: {e}")
    print("Please install with:")
    print("  pip install onnxruntime-gpu torch torchvision onnx")
    print("  or pip install onnxruntime torch torchvision onnx  # for CPU only")
    sys.exit(1)

# Add parent directory to path to import TVM utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def pytorch_to_onnx(network_name, input_shape):
    """Convert PyTorch model to ONNX format.
    
    Parameters
    ----------
    network_name : str
        Name of the network
    input_shape : tuple
        Input shape for the model
        
    Returns
    -------
    str
        Path to the ONNX model file
    """
    # Get PyTorch model  
    model_map = {
        'resnet-18': models.resnet18,
        'resnet-34': models.resnet34,
        'resnet-50': models.resnet50,
        'vgg-16': lambda: models.vgg16(num_classes=1000),
        'vgg-19': lambda: models.vgg19(num_classes=1000),
        'densenet-121': models.densenet121,
        'inception_v3': models.inception_v3,
        'mobilenet': lambda: models.mobilenet_v2(num_classes=1000),
        'squeezenet_v1.0': lambda: models.squeezenet1_0(num_classes=1000),
        'squeezenet_v1.1': lambda: models.squeezenet1_1(num_classes=1000),
    }
    
    if network_name not in model_map:
        raise ValueError(f"Unsupported network: {network_name}")
    
    model = model_map[network_name](pretrained=True)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Create temporary ONNX file
    onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
    onnx_path = onnx_file.name
    onnx_file.close()
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    
    return onnx_path

def benchmark_onnxruntime(network_name, provider='CUDAExecutionProvider', repeat=100, warmup=10):
    """Benchmark ONNX Runtime model inference performance.
    
    Parameters
    ----------
    network_name : str
        Name of the network to benchmark
    provider : str
        Execution provider ('CUDAExecutionProvider' or 'CPUExecutionProvider')
    repeat : int
        Number of inference runs for timing
    warmup : int
        Number of warmup runs before timing
        
    Returns
    -------
    tuple
        (mean_time_ms, std_time_ms) - Mean and standard deviation of inference time in milliseconds
    """
    # Determine input shape based on network
    if network_name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
    else:
        input_shape = (1, 3, 224, 224)
    
    # Convert PyTorch model to ONNX
    print(f"Converting {network_name} from PyTorch to ONNX...")
    onnx_path = pytorch_to_onnx(network_name, input_shape)
    
    try:
        # Create ONNX Runtime session
        if provider == 'CUDAExecutionProvider':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Create random input
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup runs
        print(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            _ = session.run(None, {input_name: input_data})
        
        # Actual timing runs
        print(f"Running {repeat} timing iterations...")
        times = []
        for i in range(repeat):
            start_time = time.time()
            _ = session.run(None, {input_name: input_data})
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{repeat} iterations...")
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        return mean_time, std_time
        
    finally:
        # Clean up temporary ONNX file
        try:
            os.unlink(onnx_path)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime model inference performance")
    parser.add_argument(
        "--network",
        type=str,
        choices=[
            "resnet-18", "resnet-34", "resnet-50",
            "vgg-16", "vgg-19", 
            "densenet-121",
            "inception_v3",
            "mobilenet",
            "squeezenet_v1.0", "squeezenet_v1.1",
        ],
        help="The name of neural network",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Execution provider (cuda or cpu)",
    )
    parser.add_argument("--repeat", type=int, default=100, help="Number of timing runs")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    
    args = parser.parse_args()
    
    # Map provider argument to ONNX Runtime provider names
    if args.provider == "cuda":
        # Check if CUDA is available
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            provider = 'CUDAExecutionProvider'
        else:
            print("CUDA provider not available, falling back to CPU")
            provider = 'CPUExecutionProvider'
            args.provider = "cpu"
    else:
        provider = 'CPUExecutionProvider'
    
    if args.network is None:
        networks = ["resnet-50", "mobilenet", "vgg-19", "inception_v3"]
    else:
        networks = [args.network]
    
    print("ONNX Runtime Model Benchmark")
    print("=" * 50)
    print(f"Provider: {provider}")
    print(f"Repeat: {args.repeat}")
    print(f"Warmup: {args.warmup}")
    print("=" * 50)
    print(f"{'Network':<20} {'Mean Time (ms)':<15} {'Std Dev (ms)':<15}")
    print("-" * 50)
    
    for network in networks:
        try:
            mean_time, std_time = benchmark_onnxruntime(network, provider, args.repeat, args.warmup)
            print(f"{network:<20} {mean_time:<15.2f} {std_time:<15.2f}")
        except Exception as e:
            print(f"{network:<20} {'ERROR':<15} {str(e):<15}")
    
    print("-" * 50)

if __name__ == "__main__":
    main()