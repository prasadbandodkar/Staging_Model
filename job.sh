#!/bin/bash
##SBATCH directives specify resource requirements

#SBATCH --job-name=staging_nano        # Job name
#SBATCH --time=00:15:00                 # Wall time limit (HH:MM:SS) - max 4 days for gpu queue
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --mem=2560M                      # Total memory for the job
#SBATCH --gres=gpu:1                   # Request 1 GPU (any available type)
#SBATCH --partition=gpu                # Submit to GPU partition
#SBATCH --output=runs/staging_%j.out      # Standard output file (%j = job ID)
#SBATCH --error=runs/staging_%j.err       # Standard error file

## Optional: Request specific GPU type (uncomment one if needed)
##SBATCH --gres=gpu:a100:1             # Request 1 A100 GPU (40GB)
##SBATCH --gres=gpu:rtx:1              # Request 1 Quadro RTX 6000 GPU (24GB)
##SBATCH --gres=gpu:t4:1               # Request 1 T4 GPU (16GB)

## Optional: Email notifications (uncomment and add your email)
##SBATCH --mail-type=BEGIN,END,FAIL    # Email on job start, end, and failure
##SBATCH --mail-user=your_email@tamu.edu

## Optional: Specify account for SU billing
##SBATCH --account=your_account_name

# Load required modules
# module purge                           # Clear any loaded modules
# module load GCC/11.2.0                 # Load GCC compiler (adjust version as needed)
module load CUDA/12.1.1                # Load CUDA (adjust version for your needs)
# module load cuDNN/8.4.1.50-CUDA-11.7.0 # Load cuDNN for PyTorch GPU support

# Configure uv to use scratch directory (avoid home directory quota issues)
export UV_CACHE_DIR=$SCRATCH/.uv_cache
export UV_PYTHON_INSTALL_DIR=$SCRATCH/.uv_python
export MPLCONFIGDIR=$SCRATCH/.config/matplotlib

# Optional: Load Python/PyTorch/TensorFlow module if available
# module load Python/3.9.6
# Or activate your Python virtual environment
# source $SCRATCH/venv/bin/activate

# Optional: Display GPU information
echo "Job started on $(hostname) at $(date)"
echo "GPU assigned:"
nvidia-smi

# Navigate to your working directory
cd $SCRATCH/staging/Staging_Model

# Run your training script
uv run --extra cuda121 python main.py --mode train

echo "Job finished at $(date)"
