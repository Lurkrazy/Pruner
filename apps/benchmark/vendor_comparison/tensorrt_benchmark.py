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
"""Benchmark script for native TensorRT model inference.

This script benchmarks TensorRT models directly,
providing baseline performance numbers for comparison with TVM.

NOTE: This script requires TensorRT to be installed. TensorRT is available
from NVIDIA and requires registration to download.
"""
import argparse
import time
import numpy as np
import sys
import os
import tempfile

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    import torch
    import torchvision.models as models
except ImportError as e:
    print(f"Required library not installed: {e}")
    print("TensorRT benchmark requires:")
    print("  - TensorRT (from NVIDIA Developer Portal)")
    print("  - pycuda: pip install pycuda")
    print("  - PyTorch: pip install torch torchvision")
    print("  - NVIDIA GPU with CUDA support")
    sys.exit(1)

# Add parent directory to path to import TVM utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TensorRTEngine:
    """TensorRT engine wrapper for inference."""
    
    def __init__(self, engine_path=None, engine_data=None):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        
        if engine_path:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Allocate device memory
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer to device bindings
            self.bindings.append(int(device_mem))
            
            # Append to the appropriate list
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def infer(self, input_data):
        """Run inference on the TensorRT engine."""
        # Copy input data to GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output data back to CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host']

def build_tensorrt_engine(network_name, input_shape, precision='fp32'):
    """Build TensorRT engine from PyTorch model.
    
    Parameters
    ----------
    network_name : str
        Name of the network
    input_shape : tuple
        Input shape for the model
    precision : str
        Precision mode ('fp32', 'fp16', 'int8')
        
    Returns
    -------
    bytes
        Serialized TensorRT engine
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
    model = model.cuda()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).cuda()
    
    # Export to ONNX first
    onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
    onnx_path = onnx_file.name
    onnx_file.close()
    
    try:
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
        
        # Build TensorRT engine from ONNX
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                raise RuntimeError("Failed to parse ONNX model")
        
        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
        
        # Build and serialize engine
        plan = builder.build_serialized_network(network, config)
        return plan
        
    finally:
        # Clean up ONNX file
        try:
            os.unlink(onnx_path)
        except:
            pass

def benchmark_tensorrt(network_name, precision='fp32', repeat=100, warmup=10):
    """Benchmark TensorRT model inference performance.
    
    Parameters
    ----------
    network_name : str
        Name of the network to benchmark
    precision : str
        Precision mode ('fp32', 'fp16', 'int8')
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
    
    # Build TensorRT engine
    print(f"Building TensorRT engine for {network_name} with {precision} precision...")
    engine_data = build_tensorrt_engine(network_name, input_shape, precision)
    
    # Create TensorRT engine wrapper
    engine = TensorRTEngine(engine_data=engine_data)
    
    # Create random input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup runs
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        _ = engine.infer(input_data)
    
    # Actual timing runs
    print(f"Running {repeat} timing iterations...")
    times = []
    for i in range(repeat):
        start_time = time.time()
        _ = engine.infer(input_data)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{repeat} iterations...")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time

def main():
    parser = argparse.ArgumentParser(description="Benchmark TensorRT model inference performance")
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
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="Precision mode",
    )
    parser.add_argument("--repeat", type=int, default=100, help="Number of timing runs")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    try:
        cuda.init()
        print(f"CUDA devices available: {cuda.Device.count()}")
        device = cuda.Device(0)
        print(f"Using GPU: {device.name()}")
    except:
        print("CUDA not available. TensorRT requires NVIDIA GPU.")
        sys.exit(1)
    
    if args.network is None:
        networks = ["resnet-50", "mobilenet", "vgg-19", "inception_v3"]
    else:
        networks = [args.network]
    
    print("TensorRT Model Benchmark")
    print("=" * 50)
    print(f"Precision: {args.precision}")
    print(f"Repeat: {args.repeat}")
    print(f"Warmup: {args.warmup}")
    print("=" * 50)
    print(f"{'Network':<20} {'Mean Time (ms)':<15} {'Std Dev (ms)':<15}")
    print("-" * 50)
    
    for network in networks:
        try:
            mean_time, std_time = benchmark_tensorrt(network, args.precision, args.repeat, args.warmup)
            print(f"{network:<20} {mean_time:<15.2f} {std_time:<15.2f}")
        except Exception as e:
            print(f"{network:<20} {'ERROR':<15} {str(e):<15}")
    
    print("-" * 50)

if __name__ == "__main__":
    main()