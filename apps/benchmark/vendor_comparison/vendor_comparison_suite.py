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
"""Comprehensive benchmark comparison suite for TVM vs vendor libraries.

This script compares TVM performance against multiple vendor libraries
including TensorRT, PyTorch, and ONNX Runtime on the same models and hardware.
"""
import argparse
import time
import numpy as np
import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import TVM utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import get_network

# TVM imports
import tvm
from tvm import relay, runtime
import tvm.contrib.graph_runtime as graph_runtime

# Import vendor benchmark modules
try:
    from pytorch_benchmark import benchmark_pytorch
    pytorch_available = True
except ImportError:
    pytorch_available = False

try:
    from onnxruntime_benchmark import benchmark_onnxruntime
    onnxruntime_available = True
except ImportError:
    onnxruntime_available = False

try:
    from tensorrt_benchmark import benchmark_tensorrt
    tensorrt_available = True
except ImportError:
    tensorrt_available = False

def benchmark_tvm(network_name, target, repeat=100, warmup=10):
    """Benchmark TVM model inference performance.
    
    Parameters
    ----------
    network_name : str
        Name of the network to benchmark
    target : str or tvm.target.Target
        TVM compilation target
    repeat : int
        Number of inference runs for timing
    warmup : int
        Number of warmup runs before timing
        
    Returns
    -------
    tuple
        (mean_time_ms, std_time_ms) - Mean and standard deviation of inference time in milliseconds
    """
    # Get network
    net, params, input_shape, output_shape = get_network(network_name, batch_size=1)
    
    # Compile with TVM
    print(f"Compiling {network_name} with TVM...")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(net, target=target, params=params)
    
    # Create runtime
    if str(target).startswith('cuda'):
        ctx = tvm.gpu(0)
    else:
        ctx = tvm.cpu(0)
    
    module = graph_runtime.GraphModule(lib["default"](ctx))
    data_tvm = tvm.nd.array(np.random.uniform(size=input_shape).astype("float32"), ctx)
    
    # Warmup runs
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        module.set_input("data", data_tvm)
        module.run()
        _ = module.get_output(0)
        if str(target).startswith('cuda'):
            ctx.sync()
    
    # Actual timing runs
    print(f"Running {repeat} timing iterations...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000  # Convert to milliseconds
    
    mean_time = np.mean(prof_res)
    std_time = np.std(prof_res)
    
    return mean_time, std_time

def run_framework_benchmark(framework, network, device, repeat, warmup, **kwargs):
    """Run benchmark for a specific framework.
    
    Parameters
    ----------
    framework : str
        Framework name ('tvm', 'pytorch', 'onnxruntime', 'tensorrt')
    network : str
        Network name
    device : str
        Device type ('cuda', 'cpu')
    repeat : int
        Number of timing runs
    warmup : int
        Number of warmup runs
    **kwargs : dict
        Additional framework-specific arguments
        
    Returns
    -------
    tuple or None
        (mean_time, std_time) if successful, None if failed
    """
    try:
        if framework == 'tvm':
            if device == 'cuda':
                target = tvm.target.Target("cuda")
            else:
                target = tvm.target.Target("llvm")
            return benchmark_tvm(network, target, repeat, warmup)
        
        elif framework == 'pytorch':
            if not pytorch_available:
                raise ImportError("PyTorch not available")
            return benchmark_pytorch(network, device, repeat, warmup)
        
        elif framework == 'onnxruntime':
            if not onnxruntime_available:
                raise ImportError("ONNX Runtime not available")
            return benchmark_onnxruntime(network, device, repeat, warmup)
        
        elif framework == 'tensorrt':
            if not tensorrt_available:
                raise ImportError("TensorRT not available")
            if device != 'cuda':
                raise ValueError("TensorRT only supports CUDA")
            precision = kwargs.get('precision', 'fp32')
            return benchmark_tensorrt(network, precision, repeat, warmup)
        
        else:
            raise ValueError(f"Unknown framework: {framework}")
            
    except Exception as e:
        print(f"Error benchmarking {framework}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Comprehensive TVM vs vendor library benchmark comparison")
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
        help="The name of neural network (if not specified, runs default set)",
    )
    parser.add_argument(
        "--vendors",
        type=str,
        default="pytorch,onnxruntime",
        help="Comma-separated list of vendor libraries to compare (pytorch,onnxruntime,tensorrt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run benchmarks on",
    )
    parser.add_argument("--repeat", type=int, default=100, help="Number of timing runs")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--tensorrt-precision", type=str, choices=["fp32", "fp16", "int8"], 
                       default="fp32", help="TensorRT precision mode")
    parser.add_argument("--output", type=str, help="Output JSON file to save results")
    
    args = parser.parse_args()
    
    # Parse vendor list
    vendor_list = [v.strip() for v in args.vendors.split(',')]
    frameworks = ['tvm'] + vendor_list
    
    # Check device availability
    if args.device == 'cuda':
        if not tvm.runtime.enabled("cuda") or not tvm.gpu(0).exist:
            print("CUDA not available, falling back to CPU")
            args.device = 'cpu'
    
    # Select networks
    if args.network is None:
        networks = ["resnet-50", "mobilenet", "vgg-19", "inception_v3"]
    else:
        networks = [args.network]
    
    # Check framework availability
    available_frameworks = ['tvm']
    if pytorch_available:
        available_frameworks.append('pytorch')
    if onnxruntime_available:
        available_frameworks.append('onnxruntime')
    if tensorrt_available and args.device == 'cuda':
        available_frameworks.append('tensorrt')
    
    frameworks = [f for f in frameworks if f in available_frameworks]
    
    print("TVM vs Vendor Library Benchmark Comparison")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Frameworks: {', '.join(frameworks)}")
    print(f"Networks: {', '.join(networks)}")
    print(f"Repeat: {args.repeat}")
    print(f"Warmup: {args.warmup}")
    if 'tensorrt' in frameworks:
        print(f"TensorRT Precision: {args.tensorrt_precision}")
    print("=" * 60)
    
    # Results storage
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'device': args.device,
            'frameworks': frameworks,
            'networks': networks,
            'repeat': args.repeat,
            'warmup': args.warmup,
            'tensorrt_precision': args.tensorrt_precision if 'tensorrt' in frameworks else None
        },
        'results': {}
    }
    
    # Run benchmarks
    for network in networks:
        print(f"\nBenchmarking {network}:")
        print("-" * 40)
        
        network_results = {}
        
        for framework in frameworks:
            print(f"Running {framework}...")
            
            kwargs = {}
            if framework == 'tensorrt':
                kwargs['precision'] = args.tensorrt_precision
            
            result = run_framework_benchmark(
                framework, network, args.device, args.repeat, args.warmup, **kwargs
            )
            
            if result is not None:
                mean_time, std_time = result
                network_results[framework] = {
                    'mean_time_ms': float(mean_time),
                    'std_time_ms': float(std_time)
                }
                print(f"  {framework}: {mean_time:.2f} ± {std_time:.2f} ms")
            else:
                network_results[framework] = None
                print(f"  {framework}: FAILED")
        
        results['results'][network] = network_results
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    
    # Header
    header = f"{'Network':<15}"
    for framework in frameworks:
        header += f"{framework.upper():<15}"
    print(header)
    print("-" * 60)
    
    # Results rows
    for network in networks:
        row = f"{network:<15}"
        for framework in frameworks:
            result = results['results'][network].get(framework)
            if result is not None:
                row += f"{result['mean_time_ms']:<15.2f}"
            else:
                row += f"{'FAILED':<15}"
        print(row)
    
    print("-" * 60)
    
    # Calculate speedups relative to TVM
    print("\nSPEEDUP RELATIVE TO TVM")
    print("=" * 60)
    
    header = f"{'Network':<15}"
    for framework in frameworks[1:]:  # Skip TVM itself
        header += f"{framework.upper():<15}"
    print(header)
    print("-" * 60)
    
    for network in networks:
        tvm_result = results['results'][network].get('tvm')
        if tvm_result is not None:
            tvm_time = tvm_result['mean_time_ms']
            row = f"{network:<15}"
            
            for framework in frameworks[1:]:
                vendor_result = results['results'][network].get(framework)
                if vendor_result is not None:
                    vendor_time = vendor_result['mean_time_ms']
                    speedup = tvm_time / vendor_time
                    row += f"{speedup:<15.2f}x"
                else:
                    row += f"{'N/A':<15}"
            print(row)
        else:
            print(f"{network:<15}{'TVM FAILED - cannot calculate speedups'}")
    
    print("-" * 60)
    
    # Save results to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()