#!/usr/bin/env bash


SCRIPT_DIR="$(dirname "$0")"
LOG_FILE="${SCRIPT_DIR}/acc_training.log"

echo "Starting ACCNet training in background..."
echo "Logs will be written to: ${LOG_FILE}"

nohup python3 "${SCRIPT_DIR}/acc_impl.py" > "${LOG_FILE}" 2>&1 &

echo "Training process started with PID: $!"
echo "Monitor with: tail -f ${LOG_FILE}"