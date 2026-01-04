#!/bin/bash
# CUDA environment setup for Nvidia DGX Spark with Blackwell GPUs
# Source this file before running LLaMA Factory commands

# Set CUDA library path
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH

# Optimize for Blackwell architecture
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32=1

# Disable wandb for testing (optional)
export WANDB_DISABLED=true

# Hugging Face authentication (load from .env file if exists)
if [ -f "$HOME/.env" ]; then
    export $(grep -v '^#' "$HOME/.env" | grep HF_TOKEN | xargs)
elif [ -f "$(dirname "$0")/.env" ]; then
    export $(grep -v '^#' "$(dirname "$0")/.env" | grep HF_TOKEN | xargs)
fi

# Or set your token directly here:
# export HF_TOKEN="your_token_here"

echo "✅ CUDA 13.0 environment configured for Blackwell GPUs"
echo "   LD_LIBRARY_PATH set"
echo "   PyTorch optimizations enabled"
echo ""
echo "You can now run:"
echo "  llamafactory-cli webui"
echo "  llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml"
