#!/bin/bash
##SBATCH directives specify resource requirements

#SBATCH --job-name=gpu_training        # Job name
#SBATCH --time=4:00:00                 # Wall time limit (HH:MM:SS) - max 4 days for gpu queue
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --mem=32G                      # Total memory for the job
#SBATCH --gres=gpu:1                   # Request 1 GPU (any available type)
#SBATCH --partition=gpu                # Submit to GPU partition
#SBATCH --output=gpu_train_%j.out      # Standard output file (%j = job ID)
#SBATCH --error=gpu_train_%j.err       # Standard error file

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
module purge                           # Clear any loaded modules
module load GCC/11.2.0                 # Load GCC compiler (adjust version as needed)
module load CUDA/11.8.0                # Load CUDA (adjust version for your needs)

# Optional: Load Python/PyTorch/TensorFlow module if available
# module load Python/3.9.6
# Or activate your Python virtual environment
# source $SCRATCH/venv/bin/activate

# Optional: Display GPU information
echo "Job started on $(hostname) at $(date)"
echo "GPU assigned:"
nvidia-smi

# Navigate to your working directory
cd $SCRATCH/your_project_directory

# Run your training script
python train.py --epochs 100 --batch-size 32 --gpu 0

echo "Job finished at $(date)"
