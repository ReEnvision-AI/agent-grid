#!/bin/bash

# Enable job control
set -m

cleanup() {
    echo "Shutting down the server and its child processes..."
    kill -- -"$PYTHON_PID"
    echo "Cleanup complete"
}

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please create a .env file with required variables."
    exit 1
fi

# Check if models file exists
if [ ! -f "models" ]; then
    echo "Error: models file not found. Please create a models file with a list of models."
    exit 1
fi

source models

# Display model selection menu
echo "Please select a model:"
for i in "${!MODELS[@]}"; do
    echo "$((i+1)). ${MODELS[i]}"
done

# Get user selection
while true; do
    read -p "Enter the number of your choice (1-${#MODELS[@]}): " choice
    if [[ "$choice" =~ ^[1-9][0-9]*$ && "$choice" -le "${#MODELS[@]}" ]]; then
        MODEL=${MODELS[$((choice-1))]}
        break
    else
        echo "Invalid choice. Please enter a number between 1 and ${#MODELS[@]}."
    fi
done


PORT=31331
ALLOC_TIMEOUT=6000
ATTN_CACHE_TOKENS=64000
PUBLIC_IP=$(curl ipinfo.io/ip)
P2P_FILE='./dev.id'
DISK_SPACE='120GB'
INFERENCE_MAX_LENGTH=136192

source .env

# Set a trap to call the cleanup function upon receiving SIGINT (Ctrl+C)
trap cleanup SIGINT SIGTERM EXIT

python -m agentgrid.cli.run_server \
    --public_ip $PUBLIC_IP \
    --device cuda \
    --torch_dtype bfloat16 \
    --quant_type nf4 \
    --port $PORT \
    --token $HF_TOKEN \
    --attn_cache_tokens "${ATTN_CACHE_TOKENS}" \
    --inference_max_length "${INFERENCE_MAX_LENGTH}" \
    --identity_path $P2P_FILE \
    --throughput 'eval' \
    --new_swarm \
    $MODEL &

PYTHON_PID=$!

echo "Server started with PID: $PYTHON_PID. Press Ctrl+C to stop."

wait $PYTHON_PID
