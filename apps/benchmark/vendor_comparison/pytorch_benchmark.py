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
"""Benchmark script for native PyTorch model inference.

This script benchmarks PyTorch models directly without TVM conversion,
providing baseline performance numbers for comparison with TVM.
"""
import argparse
import time
import numpy as np
import sys
import os

try:
    import torch
    import torchvision.models as models
except ImportError:
    print("PyTorch is not installed. Please install PyTorch to run this benchmark.")
    print("Install with: pip install torch torchvision")
    sys.exit(1)

# Add parent directory to path to import TVM utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import get_network

def get_pytorch_model(network_name):
    """Get PyTorch model for the given network name."""
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
    return model

def benchmark_pytorch(network_name, device='cuda', repeat=100, warmup=10):
    """Benchmark PyTorch model inference performance.
    
    Parameters
    ----------
    network_name : str
        Name of the network to benchmark
    device : str
        Device to run on ('cuda' or 'cpu')
    repeat : int
        Number of inference runs for timing
    warmup : int
        Number of warmup runs before timing
        
    Returns
    -------
    tuple
        (mean_time_ms, std_time_ms) - Mean and standard deviation of inference time in milliseconds
    """
    # Get PyTorch model
    model = get_pytorch_model(network_name)
    
    # Determine input shape based on network
    if network_name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
    else:
        input_shape = (1, 3, 224, 224)
    
    # Move model to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        device_obj = torch.device('cuda')
    else:
        device_obj = torch.device('cpu')
        device = 'cpu'
    
    # Create random input
    input_data = torch.randn(input_shape, dtype=torch.float32, device=device_obj)
    
    # Warmup runs
    print(f"Running {warmup} warmup iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    # Actual timing runs
    print(f"Running {repeat} timing iterations...")
    times = []
    with torch.no_grad():
        for i in range(repeat):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(input_data)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{repeat} iterations...")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time

def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch model inference performance")
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
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run benchmarks on",
    )
    parser.add_argument("--repeat", type=int, default=100, help="Number of timing runs")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    
    args = parser.parse_args()
    
    if args.network is None:
        networks = ["resnet-50", "mobilenet", "vgg-19", "inception_v3"]
    else:
        networks = [args.network]
    
    print("PyTorch Model Benchmark")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Repeat: {args.repeat}")
    print(f"Warmup: {args.warmup}")
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        else:
            print("CUDA requested but not available, falling back to CPU")
            args.device = 'cpu'
    print("=" * 50)
    print(f"{'Network':<20} {'Mean Time (ms)':<15} {'Std Dev (ms)':<15}")
    print("-" * 50)
    
    for network in networks:
        try:
            mean_time, std_time = benchmark_pytorch(network, args.device, args.repeat, args.warmup)
            print(f"{network:<20} {mean_time:<15.2f} {std_time:<15.2f}")
        except Exception as e:
            print(f"{network:<20} {'ERROR':<15} {str(e):<15}")
    
    print("-" * 50)

if __name__ == "__main__":
    main()