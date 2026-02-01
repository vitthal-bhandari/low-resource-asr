#!/bin/bash
# =============================================================================
# Hyak Setup Script
# Run this once to set up your environment on Hyak
# =============================================================================

set -e

echo "=========================================="
echo "Setting up Low-Resource ASR on Hyak"
echo "=========================================="

# Configuration - EDIT THESE
ACCOUNT="YOUR_ACCOUNT"              # Your Hyak account (e.g., stf, escience)
PROJECT_DIR="/gscratch/$ACCOUNT/low-resource-asr"
CONDA_DIR="/gscratch/scrubbed/$USER/miniconda3"
ENV_NAME="low-resource-asr"

echo ""
echo "Configuration:"
echo "  Account: $ACCOUNT"
echo "  Project dir: $PROJECT_DIR"
echo "  Conda dir: $CONDA_DIR"
echo ""

# =============================================================================
# Step 1: Clone or sync project
# =============================================================================

echo "Step 1: Setting up project directory..."

if [ ! -d "$PROJECT_DIR" ]; then
    echo "  Cloning repository..."
    git clone https://github.com/YOUR_USERNAME/low-resource-asr.git "$PROJECT_DIR"
else
    echo "  Project directory exists, pulling latest..."
    cd "$PROJECT_DIR"
    git pull
fi

cd "$PROJECT_DIR"

# Create necessary directories
mkdir -p logs
mkdir -p models
mkdir -p data/mozilla_speech_data

echo "  Done."
echo ""

# =============================================================================
# Step 2: Install Miniconda (if not already installed)
# =============================================================================

echo "Step 2: Setting up Miniconda..."

if [ ! -d "$CONDA_DIR" ]; then
    echo "  Downloading Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    
    echo "  Installing Miniconda to $CONDA_DIR..."
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    
    # Initialize conda
    "$CONDA_DIR/bin/conda" init bash
    source ~/.bashrc
else
    echo "  Miniconda already installed."
fi

# Source conda
source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "  Done."
echo ""

# =============================================================================
# Step 3: Create conda environment
# =============================================================================

echo "Step 3: Setting up conda environment..."

ENV_PATH="/gscratch/scrubbed/$USER/$ENV_NAME"

if [ ! -d "$ENV_PATH" ]; then
    echo "  Creating environment: $ENV_NAME..."
    conda create -p "$ENV_PATH" python=3.11 -y
else
    echo "  Environment exists."
fi

# Activate environment
conda activate "$ENV_PATH"

echo "  Done."
echo ""

# =============================================================================
# Step 4: Install dependencies
# =============================================================================

echo "Step 4: Installing dependencies..."

cd "$PROJECT_DIR"

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install transformers datasets accelerate evaluate jiwer safetensors \
    huggingface_hub pandas numpy tqdm python-dotenv requests librosa soundfile

echo "  Done."
echo ""

# =============================================================================
# Step 5: Verify GPU access
# =============================================================================

echo "Step 5: Verifying setup..."

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')

import transformers
print(f'Transformers version: {transformers.__version__}')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Copy your .env file to $PROJECT_DIR/.env"
echo "  2. Upload your data to $PROJECT_DIR/data/mozilla_speech_data/"
echo "  3. Edit the SLURM scripts in scripts/ with your account name"
echo "  4. Submit jobs with: sbatch scripts/hyak_train_all.slurm"
echo ""
echo "To activate your environment in future sessions:"
echo "  source $CONDA_DIR/etc/profile.d/conda.sh"
echo "  conda activate $ENV_PATH"
echo ""
