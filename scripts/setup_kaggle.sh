#!/bin/bash
set -euo pipefail

TALOS_DIR="${TALOS_DIR:-/kaggle/working/talos}"
BUILD_DIR="${TALOS_DIR}/build"
BINARY="Talos"
NPROC=$(nproc)

export CMAKE_PREFIX_PATH="${TALOS_DIR}/libtorch"

echo "═══ Installing build deps ═══"
apt-get update -qq
apt-get install -y -qq cmake build-essential pkg-config

echo "═══ LibTorch ═══"
LIBTORCH_DIR="${TALOS_DIR}/libtorch"
if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Downloading LibTorch (~2GB)..."
    apt-get install -y -qq wget unzip
    wget -q --show-progress \
        "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip" \
        -O /tmp/libtorch.zip
    unzip -q /tmp/libtorch.zip -d "$TALOS_DIR"
    rm -f /tmp/libtorch.zip
    echo "LibTorch extracted"
else
    echo "LibTorch already cached"
fi

echo "═══ CUDA toolkit ═══"
NVCC=$(find /usr/local/cuda* /usr/lib/cuda* /opt/cuda* -name nvcc 2>/dev/null | head -1)
if [ -z "$NVCC" ]; then
    NVCC=$(command -v nvcc 2>/dev/null || true)
fi
if [ -n "$NVCC" ]; then
    CUDA_ROOT=$(dirname "$(dirname "$NVCC")")
    echo "Found nvcc at $CUDA_ROOT"
else
    echo "CUDA toolkit not found — installing nvidia-cuda-toolkit..."
    apt-get install -y -qq nvidia-cuda-toolkit
    NVCC=$(find /usr/local/cuda* /usr/lib/cuda* /opt/cuda* -name nvcc 2>/dev/null | head -1)
    if [ -z "$NVCC" ]; then
        NVCC=$(command -v nvcc 2>/dev/null || true)
    fi
    if [ -n "$NVCC" ]; then
        CUDA_ROOT=$(dirname "$(dirname "$NVCC")")
    else
        echo "WARNING: nvcc still not found after install — trying /usr/lib/cuda"
        CUDA_ROOT="/usr/lib/cuda"
    fi
fi
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
echo "CUDA root: $CUDA_ROOT"

echo "═══ cmake configure ═══"
cd "$TALOS_DIR"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo "═══ cmake build ═══"
cmake --build "$BUILD_DIR" --config Release -j"$NPROC"

echo "═══ Build complete ═══"
