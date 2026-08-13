#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ===================================================
# Settings
# ===================================================

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODE="${MODE:-faiss}"

N_PROTO="${N_PROTO:-8}"
N_PROTO_PATCHES="${N_PROTO_PATCHES:-250000}"

N_INIT="${N_INIT:-5}"
N_ITER="${N_ITER:-50}"

IN_DIM="${IN_DIM:-1024}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1}"

# ===================================================
# Directories
# ===================================================

DATA_DIR="${DATA_DIR:-/home/stat-zx/6.TCGA-CRC/unifeats}"

CSV_DIR="${CSV_DIR:-$PROJECT_DIR/csvfiles}"
CSV_PATH="${CSV_PATH:-$CSV_DIR/train.csv}"

SAVE_DIR="${SAVE_DIR:-$PROJECT_DIR/data}"

mkdir -p "$SAVE_DIR"

# ===================================================
# Sanity Check
# ===================================================

if [ ! -d "$DATA_DIR" ]; then
    echo "[ERROR] DATA_DIR not found:"
    echo "$DATA_DIR"
    exit 1
fi

if [ ! -f "$CSV_PATH" ]; then
    echo "[ERROR] CSV_PATH not found:"
    echo "$CSV_PATH"
    exit 1
fi

# ===================================================
# Run
# ===================================================

echo "=================================================="
echo " Prototype Clustering"
echo "=================================================="
echo " Mode                 : $MODE"
echo " N Proto              : $N_PROTO"
echo " Patches per Proto    : $N_PROTO_PATCHES"
echo " Total Patches        : $((N_PROTO * N_PROTO_PATCHES))"
echo " Iterations           : $N_ITER"
echo " N Init               : $N_INIT"
echo " Feature Dim          : $IN_DIM"
echo " GPU                  : $CUDA_VISIBLE_DEVICES"
echo "=================================================="

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
python "$PROJECT_DIR/main_prototype.py" \
    --seed "$SEED" \
    --mode "$MODE" \
    --data_dir "$DATA_DIR" \
    --csv_path "$CSV_PATH" \
    --n_proto "$N_PROTO" \
    --n_proto_patches "$N_PROTO_PATCHES" \
    --n_init "$N_INIT" \
    --n_iter "$N_ITER" \
    --in_dim "$IN_DIM" \
    --num_workers "$NUM_WORKERS" \
    --save_dir "$SAVE_DIR"

echo ""
echo "=== Prototype Clustering Finished ==="