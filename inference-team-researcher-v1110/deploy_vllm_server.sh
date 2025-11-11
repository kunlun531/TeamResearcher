#!/bin/bash

export NO_PROXY="localhost,172.27.91.104,172.24.186.253,172.24.69.242,172.24.178.177,172.26.178.6,172.24.9.53,172.24.144.169,172.27.91.76,172.26.55.215,172.25.198.61,127.0.0.1,172.27.150.51,172.24.22.177,172.24.41.21,172.25.192.147,172.24.178.161,120.92.112.87,http://127.0.0.1:8265"

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "Loading environment variables from .env file..."
set -a  # automatically export all variables
source "$ENV_FILE"
set +a  # stop automatically exporting

# Validate critical variables
if [ "$MODEL_PATH" = "/your/model/path" ] || [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH not configured in .env file"
    exit 1
fi

######################################
### 1. start server           ###
######################################

echo "Starting VLLM servers..."
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6001 --disable-log-requests --gpu-memory-utilization 0.95 --max-model-len 32768 &
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6002 --disable-log-requests --gpu-memory-utilization 0.95 --max-model-len 32768 &
CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6003 --disable-log-requests --gpu-memory-utilization 0.95 --max-model-len 32768 &
CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6004 --disable-log-requests --gpu-memory-utilization 0.95 --max-model-len 32768 &
CUDA_VISIBLE_DEVICES=4 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6005 --disable-log-requests --gpu-memory-utilization 0.95 --max-model-len 32768 &
CUDA_VISIBLE_DEVICES=5 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6006 --disable-log-requests --gpu-memory-utilization 0.95 --max-model-len 32768 &
CUDA_VISIBLE_DEVICES=6 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6007 --disable-log-requests --gpu-memory-utilization 0.95 --max-model-len 32768 &
CUDA_VISIBLE_DEVICES=7 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6008 --disable-log-requests --gpu-memory-utilization 0.95 --max-model-len 32768 &

