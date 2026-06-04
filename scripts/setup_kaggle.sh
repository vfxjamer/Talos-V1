#!/bin/bash
set -uo pipefail

TALOS_DIR="${TALOS_DIR:-/kaggle/working/talos}"
BUILD_DIR="${TALOS_DIR}/build"
BINARY="Talos"
NPROC=$(nproc)

export CMAKE_PREFIX_PATH="${TALOS_DIR}/libtorch"

echo "═══ Installing build deps ═══"
apt-get update -qq 2>/dev/null || true
apt-get install -y -qq cmake build-essential pkg-config 2>&1 | tail -5
echo "OK"

echo "═══ LibTorch ═══"
LIBTORCH_DIR="${TALOS_DIR}/libtorch"
if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Downloading LibTorch (~2GB)..."
    apt-get install -y -qq wget unzip 2>&1 | tail -3
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
CUDA_ROOT=""
# Check if CUDA 12 already installed
for d in /usr/local/cuda-12* /usr/local/cuda /usr/lib/cuda-12*; do
    if [ -d "$d" ] && { [ -f "$d/include/cuda.h" ] || [ -f "$d/bin/nvcc" ]; }; then
        CUDA_ROOT="$d"
        echo "Found CUDA 12 at $d"
        break
    fi
done
# Fallback: check nvcc version
if [ -z "$CUDA_ROOT" ]; then
    NVCC=$(command -v nvcc 2>/dev/null || true)
    if [ -n "$NVCC" ]; then
        VER=$(nvcc --version 2>/dev/null | grep 'release' | grep -oP 'release \K[0-9.]+' || echo "11.5")
        MAJOR=${VER%%.*}
        if [ "$MAJOR" -ge 12 ]; then
            CUDA_ROOT=$(dirname "$(dirname "$NVCC")")
            echo "Found nvcc (CUDA $VER) at $CUDA_ROOT"
        else
            echo "Found nvcc CUDA $VER but need 12.x — will install CUDA 12.1"
        fi
    fi
fi
if [ -z "$CUDA_ROOT" ]; then
    echo "Installing CUDA 12.1 from NVIDIA..."
    # Remove conflicting CUDA 11.5 from apt first
    apt-get remove -y -qq nvidia-cuda-toolkit libcudart-dev 2>/dev/null || true
    apt-get autoremove -y -qq 2>/dev/null || true
    rm -f /usr/include/cuda.h /usr/include/cuda_runtime.h 2>/dev/null || true
    apt-get install -y -qq wget 2>&1 | tail -3
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
        -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb 2>&1 | tail -3
    apt-get update -qq 2>/dev/null || true
    apt-get install -y -qq cuda-compiler-12-1 cuda-libraries-dev-12-1 2>&1 | tail -5
    CUDA_ROOT="/usr/local/cuda-12.1"
    if [ ! -f "$CUDA_ROOT/include/cuda.h" ]; then
        # Symlink not created yet
        CUDA_ROOT="/usr/local/cuda-12.1"
        ls "$CUDA_ROOT" 2>/dev/null || echo "CUDA 12.1 dir missing"
    fi
fi
if [ -f "$CUDA_ROOT/include/cuda.h" ]; then
    export CUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT"
    export PATH="$CUDA_ROOT/bin:$PATH"
    export CUDA_VISIBLE_DEVICES="0,1"
    echo "CUDA root: $CUDA_ROOT"
    echo "nvcc: $(nvcc --version 2>&1 | grep release)"
else
    echo "WARNING: CUDA 12 toolkit not found!"
fi

echo "═══ cmake configure ═══"
rm -rf "$BUILD_DIR"
cd "$TALOS_DIR"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release 2>&1 || {
    rc=$?
    echo "cmake configure failed (exit $rc)"
    echo "CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR:-unset}"
    echo "CUDA_ROOT=${CUDA_ROOT:-unset}"
    ls /usr/local/cuda* 2>/dev/null || echo "no /usr/local/cuda*"
    ls /usr/lib/cuda* 2>/dev/null || echo "no /usr/lib/cuda*"
    command -v nvcc 2>/dev/null || echo "nvcc not found"
    exit $rc
}

echo "═══ cmake build ═══"
cmake --build "$BUILD_DIR" --config Release -j"$NPROC" 2>&1

echo "═══ Build complete ═══"
