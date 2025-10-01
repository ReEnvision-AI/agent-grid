#!/bin/bash

# Enable job control and safer scripting defaults
set -m
set -o pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
PYTHON_PID=""
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    else
        echo "Error: python3 is required but not found." >&2
        exit 1
    fi
fi

cleanup() {
    if [ -n "${PYTHON_PID:-}" ]; then
        echo "Shutting down the server and its child processes..."
        if kill -0 "$PYTHON_PID" 2>/dev/null; then
            kill "$PYTHON_PID" 2>/dev/null || true
            wait "$PYTHON_PID" 2>/dev/null || true
        fi
        echo "Cleanup complete"
    fi
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
if command -v curl >/dev/null 2>&1; then
    PUBLIC_IP=$(curl -fsS ipinfo.io/ip 2>/dev/null)
fi
if [ -z "$PUBLIC_IP" ]; then
    if command -v hostname >/dev/null 2>&1; then
        PUBLIC_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
fi
if [ -z "$PUBLIC_IP" ] && command -v ipconfig >/dev/null 2>&1; then
    PUBLIC_IP=$(ipconfig getifaddr en0 2>/dev/null)
    if [ -z "$PUBLIC_IP" ]; then
        PUBLIC_IP=$(ipconfig getifaddr en1 2>/dev/null)
    fi
fi
if [ -z "$PUBLIC_IP" ]; then
    echo "Warning: Unable to determine public IP automatically; defaulting to 127.0.0.1"
    PUBLIC_IP="127.0.0.1"
fi
P2P_FILE='./dev.id'
DISK_SPACE='120GB'
INFERENCE_MAX_LENGTH=136192

source .env

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN is not set in .env."
    exit 1
fi

# Detect best available device / dtype / quantization settings unless overridden
DETECTED_SETTINGS=$("$PYTHON_BIN" -c 'import torch
device = "cpu"
dtype = "float32"
quant = "none"
if torch.cuda.is_available():
    device = "cuda"
    dtype = "bfloat16"
    quant = "nf4"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = "mps"
    dtype = "float16"
    quant = "none"
print(device, dtype, quant)
' 2>/dev/null) || true

read -r DETECTED_DEVICE DETECTED_DTYPE DETECTED_QUANT <<<"${DETECTED_SETTINGS:-cpu float32 none}"

DEVICE=${AG_DEVICE:-$DETECTED_DEVICE}
TORCH_DTYPE=${AG_TORCH_DTYPE:-$DETECTED_DTYPE}
QUANT_TYPE=${AG_QUANT_TYPE:-$DETECTED_QUANT}

if [ "$DEVICE" != "cuda" ]; then
    QUANT_TYPE="none"
fi

echo "Using device=$DEVICE, torch_dtype=$TORCH_DTYPE, quant_type=$QUANT_TYPE"

# Set a trap to call the cleanup function upon receiving SIGINT (Ctrl+C)
trap cleanup SIGINT SIGTERM EXIT

"$PYTHON_BIN" -m agentgrid.cli.run_server \
    --public_ip "$PUBLIC_IP" \
    --device "$DEVICE" \
    --torch_dtype "$TORCH_DTYPE" \
    --quant_type "$QUANT_TYPE" \
    --port $PORT \
    --token "$HF_TOKEN" \
    --attn_cache_tokens "${ATTN_CACHE_TOKENS}" \
    --inference_max_length "${INFERENCE_MAX_LENGTH}" \
    --identity_path "$P2P_FILE" \
    --throughput 'eval' \
    --new_swarm \
    "$MODEL" &

PYTHON_PID=$!

echo "Server started with PID: $PYTHON_PID. Press Ctrl+C to stop."

wait $PYTHON_PID
