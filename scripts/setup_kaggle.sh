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
for d in /usr/local/cuda* /usr/lib/cuda* /opt/cuda*; do
    if [ -d "$d" ] && [ -f "$d/include/cuda.h" ]; then
        CUDA_ROOT="$d"
        echo "Found CUDA at $d"
        break
    fi
done
if [ -z "$CUDA_ROOT" ]; then
    NVCC=$(command -v nvcc 2>/dev/null || true)
    if [ -n "$NVCC" ]; then
        CUDA_ROOT=$(dirname "$(dirname "$NVCC")")
        echo "Found nvcc at $CUDA_ROOT"
    fi
fi
if [ -z "$CUDA_ROOT" ]; then
    echo "CUDA toolkit not found — installing nvidia-cuda-toolkit..."
    apt-get install -y nvidia-cuda-toolkit 2>&1 | tail -10
    for d in /usr/local/cuda* /usr/lib/cuda* /opt/cuda*; do
        if [ -d "$d" ] && [ -f "$d/include/cuda.h" ]; then
            CUDA_ROOT="$d"
            echo "Found CUDA after install at $d"
            break
        fi
    done
    if [ -z "$CUDA_ROOT" ]; then
        NVCC=$(command -v nvcc 2>/dev/null || true)
        if [ -n "$NVCC" ]; then
            CUDA_ROOT=$(dirname "$(dirname "$NVCC")")
            echo "Found nvcc after install at $CUDA_ROOT"
        fi
    fi
fi
if [ -z "$CUDA_ROOT" ]; then
    echo "WARNING: CUDA toolkit not found! cmake will likely fail."
    echo "Searching more broadly..."
    find / -name "cuda.h" -type f 2>/dev/null | head -5 || true
else
    export CUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT"
    export PATH="$CUDA_ROOT/bin:$PATH"
    echo "CUDA root: $CUDA_ROOT"
fi

echo "═══ cmake configure ═══"
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
