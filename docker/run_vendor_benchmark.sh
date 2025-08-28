#!/bin/bash
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

# Convenience script to build and run TVM vendor benchmark Docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TVM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="tvm-vendor-benchmark"

usage() {
    echo "Usage: $0 [build|run|help]"
    echo ""
    echo "Commands:"
    echo "  build  - Build the Docker image"
    echo "  run    - Run the Docker container interactively"
    echo "  help   - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build              # Build the image"
    echo "  $0 run                # Run with GPU support"
    echo "  CUDA_VISIBLE_DEVICES=0 $0 run  # Run with specific GPU"
    echo ""
    echo "The container includes:"
    echo "  - TVM with CUDA support"
    echo "  - PyTorch 1.8.1 with CUDA"
    echo "  - ONNX Runtime with GPU support"
    echo "  - PyCUDA (for TensorRT when manually installed)"
    echo ""
    echo "See apps/benchmark/vendor_comparison/DOCKER_GUIDE.md for full documentation."
}

build_image() {
    echo "Building TVM vendor benchmark Docker image..."
    echo "This may take 15-30 minutes for the first build."
    echo ""
    
    cd "$TVM_ROOT"
    docker build -f docker/Dockerfile.vendor_benchmark -t "$IMAGE_NAME" .
    
    echo ""
    echo "✓ Build complete!"
    echo "Run with: $0 run"
}

run_container() {
    echo "Starting TVM vendor benchmark container..."
    echo "GPU support: $(nvidia-smi &>/dev/null && echo "✓ Enabled" || echo "✗ Not available")"
    echo ""
    
    # Check if image exists
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        echo "Image $IMAGE_NAME not found. Building..."
        build_image
    fi
    
    # Determine GPU flags
    GPU_FLAGS=""
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        GPU_FLAGS="--gpus all"
        echo "Running with GPU support..."
    else
        echo "Running CPU-only (no NVIDIA GPU detected)..."
    fi
    
    # Run container
    docker run $GPU_FLAGS -it --rm \
        -v "$TVM_ROOT:/workspace" \
        "$IMAGE_NAME"
}

case "${1:-help}" in
    build)
        build_image
        ;;
    run)
        run_container
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        usage
        exit 1
        ;;
esac