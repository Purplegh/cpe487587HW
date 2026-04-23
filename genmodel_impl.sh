#!/usr/bin/env bash

set -e


# Default values

EPOCHS=50
TRAIN_RATIO=0.9
DATA_RATIO=1.0
ONNX_EVERY=10
BATCH_SIZE=128
LR=0.0002
LATENT_DIM=256
ZIP_PATH="/data/CPE_487-587/img_align_celeba.zip"
SAVE_DIR="checkpoints"
OUT_DIR="results"
NUM_WORKERS=4


# Parse command line arguments

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)       EPOCHS="$2";      shift 2 ;;
        --train_ratio)  TRAIN_RATIO="$2"; shift 2 ;;
        --data_ratio)   DATA_RATIO="$2";  shift 2 ;;
        --onnx_every)   ONNX_EVERY="$2";  shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";  shift 2 ;;
        --lr)           LR="$2";          shift 2 ;;
        --latent_dim)   LATENT_DIM="$2";  shift 2 ;;
        --zip_path)     ZIP_PATH="$2";    shift 2 ;;
        --save_dir)     SAVE_DIR="$2";    shift 2 ;;
        --out_dir)      OUT_DIR="$2";     shift 2 ;;
        --num_workers)  NUM_WORKERS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash genmodel_impl.sh [--epochs N] [--train_ratio F] [--data_ratio F] [--onnx_every N]"
            exit 1 ;;
    esac
done


# Pretty print helpers

GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RESET="\033[0m"

banner() {
    echo ""
    echo -e "${BLUE}============================================================${RESET}"
    echo -e "${BLUE}  $1${RESET}"
    echo -e "${BLUE}============================================================${RESET}"
    echo ""
}

step()  { echo -e "${GREEN}>>> $1${RESET}"; }
warn()  { echo -e "${YELLOW}WARNING: $1${RESET}"; }


# Print configuration

banner "HW04 — Generative Model Training + Inference"
echo "  Epochs       : $EPOCHS"
echo "  Data ratio   : $DATA_RATIO  ($(echo "$DATA_RATIO * 100" | bc)% of CelebA)"
echo "  Train ratio  : $TRAIN_RATIO (of selected data)"
echo "  ONNX every   : $ONNX_EVERY epochs"
echo "  Batch size   : $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Latent dim   : $LATENT_DIM"
echo "  CelebA zip   : $ZIP_PATH"
echo "  Checkpoints  : $SAVE_DIR/"
echo "  Results      : $OUT_DIR/"
echo ""


# Check dataset exists

if [ ! -f "$ZIP_PATH" ]; then
    warn "CelebA zip not found at $ZIP_PATH"
    warn "Make sure you are on the Lovelace machine."
    exit 1
fi

step "CelebA dataset found at $ZIP_PATH"


# Step 1 — Train VAE

banner "Step 1/4 — Training VAE"
python scripts/genmodel_impl.py \
    --model       vae \
    --epochs      "$EPOCHS" \
    --train_ratio "$TRAIN_RATIO" \
    --data_ratio  "$DATA_RATIO" \
    --onnx_every  "$ONNX_EVERY" \
    --batch_size  "$BATCH_SIZE" \
    --lr          "$LR" \
    --latent_dim  "$LATENT_DIM" \
    --zip_path    "$ZIP_PATH" \
    --save_dir    "$SAVE_DIR" \
    --num_workers "$NUM_WORKERS"

step "VAE training complete!"


# Step 2 — Train GAN

banner "Step 2/4 — Training GAN"
python scripts/genmodel_impl.py \
    --model       gan \
    --epochs      "$EPOCHS" \
    --train_ratio "$TRAIN_RATIO" \
    --data_ratio  "$DATA_RATIO" \
    --onnx_every  "$ONNX_EVERY" \
    --batch_size  "$BATCH_SIZE" \
    --lr          "$LR" \
    --latent_dim  "$LATENT_DIM" \
    --zip_path    "$ZIP_PATH" \
    --save_dir    "$SAVE_DIR" \
    --num_workers "$NUM_WORKERS"

step "GAN training complete!"


# Step 3 — Train Diffusion Model

banner "Step 3/4 — Training Diffusion Model"

DIFF_EPOCHS=$EPOCHS

python scripts/genmodel_impl.py \
    --model       diffusion \
    --epochs      "$DIFF_EPOCHS" \
    --train_ratio "$TRAIN_RATIO" \
    --data_ratio  "$DATA_RATIO" \
    --onnx_every  "$ONNX_EVERY" \
    --batch_size  "$BATCH_SIZE" \
    --lr          "$LR" \
    --zip_path    "$ZIP_PATH" \
    --save_dir    "$SAVE_DIR" \
    --num_workers "$NUM_WORKERS"

step "Diffusion training complete!"


# Step 4 — Inference + Metrics + Plot

banner "Step 4/4 — Inference, Metrics, and Comparison Plot"
python scripts/genmodel_infer.py \
    --save_dir   "$SAVE_DIR" \
    --out_dir    "$OUT_DIR" \
    --n_samples  25 \
    --latent_dim "$LATENT_DIM"

step "Inference complete!"


# Done

banner "All Done!"
echo "  Generated images : $OUT_DIR/vae_samples.png"
echo "                     $OUT_DIR/gan_samples.png"
echo "                     $OUT_DIR/diffusion_samples.png"
echo ""
echo "  Comparison plot  : $OUT_DIR/metrics_comparison.png"
echo "  Raw metrics CSV  : $OUT_DIR/metrics_raw.csv"
echo "  ONNX checkpoints : $SAVE_DIR/"
echo ""