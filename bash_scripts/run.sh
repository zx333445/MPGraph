#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ===================================================
# Settings
# ===================================================

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODE="${MODE:-mpgraph}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
TRAIN_BS="${TRAIN_BS:-64}"
BS="${BS:-64}"
NUM_WORKERS="${NUM_WORKERS:-12}"

# Optimizer
OPTIMIZER="${OPTIMIZER:-AdamW}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

# Directories
DATA_DIR="${DATA_DIR:-/mnt/usb/6.TCGA-CRC/unifeats}"
CSV_DIR="${CSV_DIR:-$PROJECT_DIR/csvfiles}"
TEST_CSV_PATH="${TEST_CSV_PATH:-$CSV_DIR/test.csv}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_DIR/data_cache}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/results_${MODE}}"

# Prototype
N_PROTO="${N_PROTO:-8}"
PROTO_PATH="${PROTO_PATH:-$PROJECT_DIR/data/8proto_faiss_num_2.5e+05.pkl}"

# ===================================================

mkdir -p "$RESULTS_DIR" "$CACHE_DIR"

echo "=================================================="
echo " Project Root : $PROJECT_DIR"
echo " Running Mode : $MODE"
echo " Optimizer    : $OPTIMIZER"
echo " LR           : $LR"
echo " Weight Decay : $WEIGHT_DECAY"
echo " Epochs       : $NUM_EPOCHS"
echo " Train BS     : $TRAIN_BS"
echo " Eval BS      : $BS"
echo " GPU          : $CUDA_VISIBLE_DEVICES"
echo " Num Proto    : $N_PROTO"
echo " Proto Path   : $PROTO_PATH"
echo "=================================================="

# Check required paths
if [ ! -d "$DATA_DIR" ]; then
    echo "[ERROR] DATA_DIR does not exist: $DATA_DIR"
    exit 1
fi

if [ ! -f "$TEST_CSV_PATH" ]; then
    echo "[ERROR] TEST_CSV_PATH does not exist: $TEST_CSV_PATH"
    exit 1
fi

if [ "$MODE" = "mpgraph" ] || [ "$MODE" = "panther" ] || [ "$MODE" = "h2t" ]; then
    if [ ! -f "$PROTO_PATH" ]; then
        echo "[ERROR] Prototype file does not exist: $PROTO_PATH"
        exit 1
    fi
fi

EXTRA_ARGS=()

if [ "$MODE" = "mpgraph" ] || [ "$MODE" = "panther" ] || [ "$MODE" = "h2t" ]; then
    EXTRA_ARGS+=(--n_proto "$N_PROTO")
    EXTRA_ARGS+=(--proto_path "$PROTO_PATH")
fi

# ===================================================
# Five-fold training
# ===================================================

for FOLD in {1..5}; do

    echo ""
    echo ">>>>>>>>>>>>>>>> Running Fold $FOLD <<<<<<<<<<<<<<<<"

    TRAIN_CSV_PATH="$CSV_DIR/fold${FOLD}/train.csv"
    VAL_CSV_PATH="$CSV_DIR/fold${FOLD}/val.csv"

    LOG_DIR="$RESULTS_DIR/fold${FOLD}/logs"
    SAVE_MODEL_PATH="$RESULTS_DIR/fold${FOLD}/f${FOLD}.pth"

    if [ ! -f "$TRAIN_CSV_PATH" ]; then
        echo "[ERROR] Training CSV not found: $TRAIN_CSV_PATH"
        exit 1
    fi

    if [ ! -f "$VAL_CSV_PATH" ]; then
        echo "[ERROR] Validation CSV not found: $VAL_CSV_PATH"
        exit 1
    fi

    mkdir -p "$LOG_DIR"

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    python "$PROJECT_DIR/train.py" \
        --mode "$MODE" \
        --data_dir "$DATA_DIR" \
        --train_csv_path "$TRAIN_CSV_PATH" \
        --val_csv_path "$VAL_CSV_PATH" \
        --test_csv_path "$TEST_CSV_PATH" \
        --cache_dir "$CACHE_DIR" \
        --fold "$FOLD" \
        --train_batch_size "$TRAIN_BS" \
        --batch_size "$BS" \
        --num_workers "$NUM_WORKERS" \
        --num_epochs "$NUM_EPOCHS" \
        --optimizer "$OPTIMIZER" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --save_model_path "$SAVE_MODEL_PATH" \
        --logdir "$LOG_DIR" \
        --use_tensorboard \
        "${EXTRA_ARGS[@]}"

    echo ">>>>>>>>>>>>>>>> Fold $FOLD Finished <<<<<<<<<<<<<<<<"

done

echo ""
echo "=== All Folds Finished Successfully! ==="