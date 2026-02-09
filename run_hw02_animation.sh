#!/bin/bash

echo "Starting HW02 animation run..."
echo "Started at: $(date)"

nohup python scripts/binaryclassification_animate_impl.py \
    > hw02_animation.log 2>&1 &

echo "Process running in background."
echo "Logs are being written to hw02_animation.log"

