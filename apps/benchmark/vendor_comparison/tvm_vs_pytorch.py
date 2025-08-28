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
"""Direct comparison between TVM and PyTorch on the same models."""
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vendor_comparison_suite import run_framework_benchmark
import tvm

def main():
    parser = argparse.ArgumentParser(description="Compare TVM vs PyTorch performance")
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
        default="cuda",
        help="Device to run benchmarks on",
    )
    parser.add_argument("--repeat", type=int, default=100, help="Number of timing runs")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda':
        if not tvm.runtime.enabled("cuda") or not tvm.gpu(0).exist:
            print("CUDA not available, falling back to CPU")
            args.device = 'cpu'
    
    if args.network is None:
        networks = ["resnet-50", "mobilenet", "vgg-19", "inception_v3"]
    else:
        networks = [args.network]
    
    print("TVM vs PyTorch Performance Comparison")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Repeat: {args.repeat}")
    print(f"Warmup: {args.warmup}")
    print("=" * 50)
    print(f"{'Network':<20} {'TVM (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for network in networks:
        # Benchmark TVM
        tvm_result = run_framework_benchmark('tvm', network, args.device, args.repeat, args.warmup)
        
        # Benchmark PyTorch
        pytorch_result = run_framework_benchmark('pytorch', network, args.device, args.repeat, args.warmup)
        
        if tvm_result is not None and pytorch_result is not None:
            tvm_time, tvm_std = tvm_result
            pytorch_time, pytorch_std = pytorch_result
            speedup = pytorch_time / tvm_time
            
            print(f"{network:<20} {tvm_time:<15.2f} {pytorch_time:<15.2f} {speedup:<10.2f}x")
        else:
            tvm_str = f"{tvm_result[0]:.2f}" if tvm_result else "FAILED"
            pytorch_str = f"{pytorch_result[0]:.2f}" if pytorch_result else "FAILED"
            print(f"{network:<20} {tvm_str:<15} {pytorch_str:<15} {'N/A':<10}")
    
    print("-" * 50)

if __name__ == "__main__":
    main()