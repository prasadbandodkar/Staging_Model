# Quick Installation Guide

This is a streamlined installation guide. For detailed documentation, see [README.md](README.md).

## Prerequisites

- **No system Python required** - uv will manage Python 3.11 automatically
- For GPU acceleration:
  - Linux: NVIDIA GPU with CUDA support
  - macOS: Apple Silicon (M1/M2/M3/M4) for MPS support

## Installation Steps

### 1. Clone and Navigate

```bash
git clone <your-repo-url>
cd Staging_Model
```

### 2. Create Virtual Environment

**Using uv (recommended):**

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv in current directory
uv venv .venv --python 3.11

# Activate it
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

**Using standard venv:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

### 3. Install PyTorch (choose your platform)

**Linux with NVIDIA GPU (CUDA 12.1):**

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Linux with NVIDIA GPU (CUDA 11.8):**

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**macOS (Apple Silicon with MPS or Intel):**

```bash
uv pip install torch torchvision
```

### 4. Install Project Dependencies

```bash
uv pip install -e .
```

### 5. Verify Installation

```bash
python verify_installation.py
```

This will check:
- ✓ Python version
- ✓ PyTorch installation
- ✓ GPU acceleration (CUDA/MPS/CPU)
- ✓ All dependencies

## Quick Start

Once installed, train a model:

```bash
python main.py --mode train
```

Monitor with TensorBoard:

```bash
python main.py --mode tensorboard
```

## Troubleshooting

### "Command not found: uv pip"

Make sure your virtual environment is activated:

```bash
source .venv/bin/activate  # macOS/Linux
```

### "No module named 'torch'"

Your virtual environment isn't activated or PyTorch wasn't installed. Activate the environment and repeat step 3.

### Training is slow (no GPU detected)

Check GPU availability:

```bash
# For NVIDIA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# For Apple Silicon
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

If GPU is available but not detected, reinstall PyTorch with the correct index URL for your platform.

## Complete Workflow Example (Linux with CUDA 12.1)

```bash
# Clone
git clone <your-repo-url>
cd Staging_Model

# Setup environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install PyTorch with CUDA
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
uv pip install -e .

# Verify
python verify_installation.py

# Start training
python main.py --mode train
```

## Complete Workflow Example (macOS with Apple Silicon)

```bash
# Clone
git clone <your-repo-url>
cd Staging_Model

# Setup environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install PyTorch (with MPS support)
uv pip install torch torchvision

# Install dependencies
uv pip install -e .

# Verify
python verify_installation.py

# Start training
python main.py --mode train
```

---

For more details, see [README.md](README.md)
