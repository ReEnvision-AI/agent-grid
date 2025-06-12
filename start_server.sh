#!/bin/bash

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please create a .env file with required variables."
    exit 1
fi

# Check if models file exists
if [ ! -f "models" ]; then
    echo "Error: models file not found. Please create a models.sh file with a list of models."
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
    if [[ "$choice" =~ ^[1-9]$ ]]; then
        MODEL=${MODELS[$((choice-1))]}
        break
    else
        echo "Invalid choice. Please enter a number between 1 and ${#MODELS[@]}."
    fi
done


PORT=31331
MAX_LENGTH=72192
ALLOC_TIMEOUT=6000
ATTN_CACHE_TOKENS=64000
PUBLIC_IP=$(curl ipinfo.io/ip)
P2P_FILE='./dev.id'
DISK_SPACE='120GB'
#ACCELERATE_INIT_INCLUDE_BUFFERS=FALSE

source .env

python -m agentgrid.cli.run_server \
    --max_disk_space $DISK_SPACE \
    --public_ip $PUBLIC_IP \
    --port $PORT \
    --token $HF_TOKEN \
    --identity_path $P2P_FILE \
    --throughput 'eval' \
    --inference_max_length "${MAX_LENGTH}" \
    --attn_cache_tokens "${ATTN_CACHE_TOKENS}" \
    $MODEL
