#!/bin/bash

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
    if [[ "$choice" =~ ^[1-9]$ ]]; then
        MODEL=${MODELS[$((choice-1))]}
        break
    else
        echo "Invalid choice. Please enter a number between 1 and ${#MODELS[@]}."
    fi
done


PORT=31331
MAX_LENGTH=136192
ALLOC_TIMEOUT=6000
ATTN_CACHE_TOKENS=128000
#ATTN_CACHE_TOKENS=32768
PUBLIC_IP=$(curl ipinfo.io/ip)
P2P_FILE='./dev.id'
DISK_SPACE='120GB'
INFERENCE_MAX_LENGTH=40960
#ACCELERATE_INIT_INCLUDE_BUFFERS=FALSE
MAX_CHUNK_SIZE_BYTES=1073741824

source .env

python -m agentgrid.cli.run_server \
    --max_disk_space $DISK_SPACE \
    --public_ip $PUBLIC_IP \
    --device cuda \
    --torch_dtype bfloat16 \
    --quant_type nf4 \
    --port $PORT \
    --max_chunk_size_bytes "${MAX_CHUNK_SIZE_BYTES}" \
    --token $HF_TOKEN \
    --attn_cache_tokens "${ATTN_CACHE_TOKENS}" \
    --inference_max_length "${INFERENCE_MAX_LENGTH}" \
    --identity_path $P2P_FILE \
    --throughput 'eval' \
    --new_swarm \
    $MODEL
