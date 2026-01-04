# LLaMA Factory Setup Guide for Nvidia DGX Spark (ARM64 + Blackwell GPU)

This guide provides step-by-step instructions for setting up LLaMA Factory on an **Nvidia DGX Spark** with **ARM64 (Grace CPU) + Blackwell GPU architecture** and **CUDA 13.0**.

## System Requirements

- **Hardware**: Nvidia DGX Spark with Grace CPU (ARM64) and Blackwell GB10 GPU
- **CUDA Version**: 13.0
- **Python**: 3.9+
- **Package Manager**: uv (recommended for speed and isolation)

---

## Installation Steps

### 1. Clone the Repository

```bash
cd ~/dev/ai_projects/
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

### 2. Create Isolated Virtual Environment

**Using uv (recommended):**

```bash
# Create virtual environment (creates .venv directory)
uv venv

# Activate virtual environment
source .venv/bin/activate
```

**Alternative using venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> **Important**: The virtual environment ensures all packages are installed in `.venv/` directory, **NOT in your system Python**. This prevents contamination of root installations.

### 3. Install PyTorch with CUDA 13.0 Support

**Critical for Blackwell GPUs** - Install PyTorch with CUDA 13.0:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Install PyTorch with CUDA 13.0 (matches your system)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**Verify installation:**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.9.1+cu130
CUDA available: True
CUDA version: 13.0
```

### 4. Install LLaMA Factory

```bash
# Still in activated virtual environment
uv pip install -e ".[metrics,deepspeed]"
```

This installs:
- Core LLaMA Factory package (editable mode)
- Metrics dependencies (BLEU, ROUGE, etc.)
- DeepSpeed for distributed training

### 5. Setup CUDA Environment

The `setup_cuda_env.sh` script is already created in the project root. Review it:

```bash
cat setup_cuda_env.sh
```

Contents:
```bash
#!/bin/bash
# CUDA environment setup for Nvidia DGX Spark with Blackwell GPUs

# Set CUDA library path
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH

# Optimize for Blackwell architecture
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32=1

# Disable wandb for testing (optional)
export WANDB_DISABLED=true

echo "✅ CUDA 13.0 environment configured for Blackwell GPUs"
```

### 6. Verify Complete Setup

```bash
# Activate environment and setup CUDA
source .venv/bin/activate
source setup_cuda_env.sh

# Run comprehensive test
python -c "
import torch
print('='*60)
print('LLaMA Factory Setup Verification')
print('='*60)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
print('='*60)
print('✅ Setup complete and ready for training!')
"
```

Expected output:
```
============================================================
LLaMA Factory Setup Verification
============================================================
PyTorch version: 2.9.1+cu130
CUDA available: True
CUDA version: 13.0
GPU count: 1
GPU name: NVIDIA GB10
GPU memory: 119.70 GB
BF16 supported: True
Compute capability: (12, 1)
============================================================
✅ Setup complete and ready for training!
```

---

## Daily Usage

### Starting a Session

**Every time you want to use LLaMA Factory:**

```bash
# Navigate to project
cd ~/dev/ai_projects/LLaMA-Factory

# Activate virtual environment
source .venv/bin/activate

# Setup CUDA environment
source setup_cuda_env.sh

# Now run your commands
```

**One-liner for convenience:**

```bash
cd ~/dev/ai_projects/LLaMA-Factory && source .venv/bin/activate && source setup_cuda_env.sh
```

### Running the Web UI

```bash
source .venv/bin/activate && source setup_cuda_env.sh
llamafactory-cli webui
```

Access at: `http://localhost:7860` or `http://<your-dgx-ip>:7860`

### Running CLI Training

```bash
source .venv/bin/activate && source setup_cuda_env.sh
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

### Multi-GPU Training

For training across multiple GPUs on your DGX Spark:

```bash
source .venv/bin/activate && source setup_cuda_env.sh

# Check available GPUs
nvidia-smi

# Train on all GPUs (adjust NPROC_PER_NODE based on your GPU count)
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

---

## Verification of Isolation

### Check Installation Locations

```bash
# With virtual environment ACTIVE
source .venv/bin/activate
which python
# Expected: /home/username/dev/ai_projects/LLaMA-Factory/.venv/bin/python

# Check package count in virtual environment
uv pip list | wc -l
# Should show 200+ packages

# DEACTIVATE virtual environment
deactivate

# Check system Python
which python
# Expected: /usr/bin/python

# Check system packages (should NOT have torch, transformers, etc.)
python -m pip list | grep -E "(torch|transformers|llamafactory)"
# Expected: No output (packages not in system Python)
```

### Package Locations

- **Virtual environment packages**: `.venv/lib/python3.12/site-packages/`
- **System Python packages**: `/usr/lib/python3/dist-packages/` (unchanged)

All LLaMA Factory dependencies are isolated in the `.venv/` directory and will **never affect** your system Python installation.

---

## Optional: Flash Attention 2

Flash Attention 2 provides significant performance improvements for Blackwell GPUs but may have compatibility issues with ARM64 + CUDA 13.0.

### Attempt Installation

```bash
source .venv/bin/activate && source setup_cuda_env.sh

# Install build dependencies
uv pip install wheel ninja packaging

# Try to install Flash Attention
uv pip install flash-attn --no-build-isolation
```

### If Flash Attention Fails

**Option 1**: Skip it - LLaMA Factory works fine without Flash Attention, just slightly slower.

**Option 2**: Build from source (time-consuming):

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
source ../setup_cuda_env.sh
MAX_JOBS=8 uv pip install . --no-build-isolation
```

**Note**: Flash Attention is an optimization, not a requirement. Most users can skip this.

---

## Optional: vLLM for Fast Inference

vLLM provides accelerated inference but requires additional setup:

```bash
source .venv/bin/activate && source setup_cuda_env.sh
uv pip install vllm
```

Then use with:

```bash
export VLLM_USE_MODELSCOPE=False
llamafactory-cli api --backend vllm
```

---

## Troubleshooting

### Issue: "CUDA not available"

**Solution**: Make sure you source the CUDA environment script:

```bash
source setup_cuda_env.sh
```

### Issue: "Your setup doesn't support bf16/gpu"

**Cause**: PyTorch CPU version installed or CUDA environment not configured.

**Solution**:

1. Verify PyTorch has CUDA:
   ```bash
   python -c "import torch; print(torch.__version__)"
   # Should show: 2.9.1+cu130 (not +cpu)
   ```

2. If it shows `+cpu`, reinstall:
   ```bash
   uv pip uninstall torch torchvision torchaudio
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   ```

3. Source CUDA environment:
   ```bash
   source setup_cuda_env.sh
   ```

### Issue: Training crashes with "out of memory"

**Solution**: Reduce batch size or use gradient accumulation in your YAML config:

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

### Issue: Compute capability warning (12.1 vs 12.0)

This is expected for Blackwell GPUs and can be safely ignored. PyTorch 2.9.1 officially supports up to compute capability 12.0, but your Blackwell GB10 has 12.1. Training will still work.

---

## Cleanup and Uninstallation

### Remove Everything (Project and Dependencies)

```bash
# Navigate to parent directory
cd ~/dev/ai_projects/

# Delete the entire project (includes .venv)
rm -rf LLaMA-Factory
```

This removes **all** installed packages. Your system Python remains untouched.

### Keep Project, Remove Virtual Environment Only

```bash
cd ~/dev/ai_projects/LLaMA-Factory
rm -rf .venv
```

To recreate later:

```bash
uv venv
source .venv/bin/activate
# Re-run installation steps
```

---

## Additional Resources

- **LLaMA Factory Documentation**: [README.md](README.md)
- **Training Examples**: `examples/train_lora/`, `examples/train_full/`
- **Dataset Configuration**: `data/dataset_info.json`
- **Example Configs**: `examples/train_lora/*.yaml`
- **Project Guidelines**: `.github/copilot-instructions.md`

---

## Quick Reference Commands

```bash
# Daily startup (combine all activation steps)
cd ~/dev/ai_projects/LLaMA-Factory && source .venv/bin/activate && source setup_cuda_env.sh

# Launch Web UI
llamafactory-cli webui

# Train with example config
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

# Chat with trained model
llamafactory-cli chat --model_name_or_path saves/qwen3-4b/lora/sft

# Export/merge LoRA adapter
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml

# Run tests
make test

# Check code quality
make quality

# Format code
make style
```

---

## Notes for DGX Spark Specifics

### ARM64 Architecture Considerations

- Most Python packages work on ARM64, but some may need compilation from source
- PyTorch provides native ARM64 wheels for CUDA 13.0
- If a package fails to install, check for ARM64-specific wheels or build from source

### Blackwell GPU Features

- **Compute Capability**: 12.1 (cutting-edge, may have limited support in some libraries)
- **Memory**: 120 GB HBM3e (excellent for large models)
- **BF16**: Fully supported (use `bf16: true` in training configs)
- **TF32**: Enabled automatically via `setup_cuda_env.sh`

### Multi-GPU Setup

If your DGX Spark has multiple GPUs:

```bash
# Check GPU topology
nvidia-smi topo -m

# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Train with DeepSpeed ZeRO-3 for multi-GPU
llamafactory-cli train examples/train_lora/qwen3_lora_sft_ds3.yaml
```

---

**Last Updated**: January 3, 2026  
**DGX Spark Configuration**: ARM64 Grace CPU + Blackwell GB10 GPU, CUDA 13.0  
**Tested PyTorch Version**: 2.9.1+cu130
