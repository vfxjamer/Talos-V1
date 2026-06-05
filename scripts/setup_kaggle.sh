#!/bin/bash
set -uo pipefail

export DEBIAN_FRONTEND=noninteractive

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
# Remove old CUDA 11.5 that conflicts with 12.x
apt-get remove -y -qq nvidia-cuda-toolkit libcudart-dev 2>/dev/null || true
apt-get autoremove -y -qq 2>/dev/null || true
# Kill leftover nvcc wrapper that points to old CUDA
rm -f /usr/bin/nvcc /usr/lib/nvidia-cuda-toolkit/bin/nvcc 2>/dev/null || true
rm -f /usr/include/cuda.h /usr/include/cuda_runtime.h 2>/dev/null || true

CUDA_ROOT=""
# Check CUDA in priority order (Kaggle's full install first)
for d in /usr/local/cuda-12 /usr/local/cuda /usr/local/cuda-12.1 /usr/lib/cuda-12; do
    if [ -d "$d" ] && [ -f "$d/bin/nvcc" ]; then
        VER=$("$d/bin/nvcc" --version 2>/dev/null | grep 'release' | grep -oP 'release \K[0-9.]+' || echo "0")
        MAJOR=${VER%%.*}
        if [ "$MAJOR" -ge 12 ]; then
            CUDA_ROOT="$d"
            echo "Found CUDA $VER at $d"
            break
        fi
    fi
done
if [ -z "$CUDA_ROOT" ]; then
    echo "Installing CUDA 12.1 from NVIDIA..."
    apt-get install -y -qq wget 2>&1 | tail -3
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
        -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb 2>&1 | tail -3
    apt-get update -qq 2>/dev/null || true
    apt-get install -y -qq cuda-toolkit-12-1 2>&1 | tail -5
    CUDA_ROOT="/usr/local/cuda-12.1"
fi
if [ -n "$CUDA_ROOT" ]; then
    # Force-clean any CUDA 11.5 headers that cmake finds via /usr/include
    find /usr/include -name 'cuda*' -delete 2>/dev/null || true
    find /usr/include -name 'nvtx*' -delete 2>/dev/null || true
    rm -rf /usr/include/crt 2>/dev/null || true
    # NVIDIA repo keyring (needed for extra packages)
    if [ ! -f /etc/apt/sources.list.d/cuda*.sources ] 2>/dev/null; then
        apt-get install -y -qq wget 2>&1 | tail -1
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -O /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb 2>&1 | tail -1
        apt-get update -qq 2>/dev/null || true
    fi
    # Install packages that LibTorch needs (nvtx, cublas)
    apt-get install -y -qq cuda-nvtx-12-1 cuda-cublas-dev-12-1 2>&1 | tail -5 || true
    # nvToolsExt.h may be at nvtx3/ symlink it if needed
    if [ ! -f "$CUDA_ROOT/include/nvToolsExt.h" ]; then
        NVTX_H=$(find "$CUDA_ROOT" -name "nvToolsExt.h" 2>/dev/null | head -1)
        if [ -n "$NVTX_H" ]; then
            ln -sf "$NVTX_H" "$CUDA_ROOT/include/nvToolsExt.h"
            echo "Symlinked nvToolsExt.h"
        else
            # Search system wide
            NVTX_H=$(find /usr/local -name "nvToolsExt.h" 2>/dev/null | head -1)
            if [ -n "$NVTX_H" ]; then
                ln -sf "$NVTX_H" "$CUDA_ROOT/include/nvToolsExt.h"
                echo "Symlinked nvToolsExt.h from $NVTX_H"
            fi
        fi
    fi
    export CUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT"
    export PATH="$CUDA_ROOT/bin:$PATH"
    export CUDA_VISIBLE_DEVICES="0,1"
    echo "CUDA root: $CUDA_ROOT"
    echo "nvcc: $(nvcc --version 2>&1 | grep release)"
else
    echo "WARNING: CUDA 12 toolkit not found!"
fi

echo "═══ cmake configure ═══"
cd "$TALOS_DIR"
if [ ! -f "$BUILD_DIR/build.ninja" ] && [ ! -f "$BUILD_DIR/Makefile" ]; then
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
else
    echo "Build directory exists, reconfigure if needed"
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
fi

echo "═══ cmake build ═══"
cmake --build "$BUILD_DIR" --config Release -j"$NPROC" 2>&1

echo "═══ Build complete ═══"
