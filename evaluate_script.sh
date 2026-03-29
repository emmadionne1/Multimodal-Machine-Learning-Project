#!/bin/bash
#SBATCH --job-name=eval_chartqa
#SBATCH --partition=general
#SBATCH --output=slurm/eval_chartqa_%j.out
#SBATCH --error=slurm/eval_chartqa_%j.err
#SBATCH --export=ALL
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --ntasks=1

# Create slurm log directory if it doesn't exist
mkdir -p slurm

# Environment setup
source "$HOME/.bashrc"
conda init
conda activate bsampling
cd "$HOME/Multimodal-Machine-Learning-Project"

# HuggingFace Cache variables
export HF_HOME="/data/user_data/bsood/.hf_cache"
export HF_HUB_CACHE="/data/user_data/bsood/.hf_cache/hub"
export HF_DATASETS_CACHE="/data/user_data/bsood/.hf_cache/datasets"

# Path to the specific checkpoint you wanted to evaluate
PROJECTOR_WEIGHTS="/home/bsood/Multimodal-Machine-Learning-Project/outputs/bhavesh-epoch3 checkpoint-1770/projector.pt"

echo "Starting ChartQA Evaluation"
echo "Using Projector Weights: $PROJECTOR_WEIGHTS"
nvidia-smi

python3 evaluate_checkpoint.py \
    --projector_weights "$PROJECTOR_WEIGHTS" \
    --max_new_tokens 32

echo "Job finished."