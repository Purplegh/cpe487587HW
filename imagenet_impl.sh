#!/bin/env bash
# imagenet_impl.sh
# Runs ImageNet CNN training in the background.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nohup python -u "${SCRIPT_DIR}/scripts/imagenet_impl.py" \
    --epochs 10000\
    --train_ratio 0.005 \
    --val_ratio 0.005 \
    > "${SCRIPT_DIR}/scripts/training.log" 2>&1 &

echo "Training started in background with PID: $!"
echo "Monitor progress: tail -f ${SCRIPT_DIR}/scripts/training.log"