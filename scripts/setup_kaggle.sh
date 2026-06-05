#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
APT_OPTS="-o DPkg::Lock::Timeout=120 -qq"

# Kill any apt/dpkg processes left from a previous killed session
pkill -9 apt-get 2>/dev/null || true
pkill -9 dpkg 2>/dev/null || true
rm -f /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/cache/apt/archives/lock 2>/dev/null || true
dpkg --configure -a 2>/dev/null || true

TALOS_DIR="${TALOS_DIR:-/kaggle/working/talos}"
BUILD_DIR="${TALOS_DIR}/build"
BINARY="Talos"
NPROC=$(nproc)

export CMAKE_PREFIX_PATH="${TALOS_DIR}/libtorch"

echo "═══ Installing build deps ═══"
apt-get update $APT_OPTS 2>/dev/null || true
apt-get install $APT_OPTS --no-install-recommends cmake build-essential pkg-config 2>&1 | tail -5
echo "OK"

echo "═══ LibTorch ═══"
LIBTORCH_DIR="${TALOS_DIR}/libtorch"
if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Downloading LibTorch (~2GB)..."
    apt-get install $APT_OPTS --no-install-recommends wget unzip 2>&1 | tail -3
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
# Kaggle T4x2 already has CUDA 12.1 at /usr/local/cuda-12
CUDA_ROOT=""
for d in /usr/local/cuda-12 /usr/local/cuda /usr/local/cuda-12.1; do
    if [ -d "$d" ] && [ -f "$d/bin/nvcc" ]; then
        CUDA_ROOT="$d"
        echo "Found CUDA at $d"
        break
    fi
done

if [ -n "$CUDA_ROOT" ]; then
    # Remove conflicting old CUDA headers
    find /usr/include -name 'cuda*' -delete 2>/dev/null || true
    find /usr/include -name 'nvtx*' -delete 2>/dev/null || true
    rm -rf /usr/include/crt 2>/dev/null || true

    export CUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT"
    export PATH="$CUDA_ROOT/bin:$PATH"
    export CUDA_VISIBLE_DEVICES="0,1"
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
        exit $rc
    }
else
    echo "Reconfiguring..."
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
fi

echo "═══ cmake build ═══"
cmake --build "$BUILD_DIR" --config Release -j"$NPROC" 2>&1

echo "═══ Build complete ═══"
