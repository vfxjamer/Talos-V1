#!/bin/bash
# Talos v1 Kaggle Dataset Packager
# Run on Linux/WSL AFTER cross-compiling the binary for Linux.
#
# Usage:
#   1. Build Talos binary for Linux x86_64 (CUDA):
#      mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DRG_CUDA_SUPPORT=ON && make -j
#   2. Run this script from the repo root:
#      bash Kaggle/package_dataset.sh
#
# Output: talos-training.tar.gz  (upload to Kaggle Datasets)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="/tmp/talos-dataset"
OUTPUT_TAR="$REPO_ROOT/talos-training.tar.gz"

echo "=== Talos v1 Dataset Packager ==="

# ── 1. Binary ────────────────────────────────────────────────
BINARY_SRC="$REPO_ROOT/build/Talos"
if [ ! -f "$BINARY_SRC" ]; then
    echo "ERROR: Binary not found at $BINARY_SRC"
    echo "Build it first with:"
    echo "  mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DRG_CUDA_SUPPORT=ON && make -j"
    exit 1
fi

# ── 2. Collision meshes ─────────────────────────────────────
MESHES_SRC="$REPO_ROOT/collision_meshes"
if [ ! -d "$MESHES_SRC" ]; then
    echo "ERROR: collision_meshes/ not found at $MESHES_SRC"
    echo "Copy collision_meshes/ from a RocketSim build"
    exit 1
fi

# ── 3. LibTorch (CUDA) ──────────────────────────────────────
# Download if not present
LIBTORCH_DIR="/tmp/libtorch-cuda"
if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Downloading LibTorch (CUDA 12.1)..."
    wget -q --show-progress \
        https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip \
        -O /tmp/libtorch.zip
    unzip -q /tmp/libtorch.zip -d /tmp/
    mv /tmp/libtorch "$LIBTORCH_DIR"
    rm /tmp/libtorch.zip
fi

# ── 4. Replay data ──────────────────────────────────────────
REPLAY_SRC="$REPO_ROOT/serialized_replays.bin"
if [ ! -f "$REPLAY_SRC"]; then
    echo "WARNING: serialized_replays.bin not found at $REPLAY_SRC"
    echo "No replay-based state initialization will be available."
    echo "Run scripts/parse_replays.py first to create it."
    REPLAY_SRC=""
fi

# ── Assemble ─────────────────────────────────────────────────
echo "Assembling dataset in $OUTPUT_DIR..."
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

cp "$BINARY_SRC" "$OUTPUT_DIR/Talos"
chmod +x "$OUTPUT_DIR/Talos"
echo "  Binary: $(ls -lh "$OUTPUT_DIR/Talos" | awk '{print $5}')"

cp -r "$MESHES_SRC" "$OUTPUT_DIR/collision_meshes"
echo "  Meshes: $(du -sh "$OUTPUT_DIR/collision_meshes" | cut -f1)"

cp -r "$LIBTORCH_DIR" "$OUTPUT_DIR/libtorch"
echo "  LibTorch: $(du -sh "$OUTPUT_DIR/libtorch" | cut -f1)"

if [ -n "$REPLAY_SRC" ]; then
    cp "$REPLAY_SRC" "$OUTPUT_DIR/serialized_replays.bin"
    echo "  Replays: $(ls -lh "$OUTPUT_DIR/serialized_replays.bin" | awk '{print $5}')"
fi

# ── Tar ──────────────────────────────────────────────────────
echo "Creating $OUTPUT_TAR..."
cd /tmp
tar czf "$OUTPUT_TAR" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"

echo ""
echo "=== Done ==="
echo "Upload $OUTPUT_TAR to Kaggle Datasets:"
echo "  https://www.kaggle.com/datasets/YOUR_USERNAME/talos-training/create"
echo ""
echo "Dataset contents:"
ls -lh "$OUTPUT_DIR/"
echo ""
echo "Archive size: $(ls -lh "$OUTPUT_TAR" | awk '{print $5}')"
