#!/bin/bash
#SBATCH --job-name=finetune_chartqa
#SBATCH --partition=general
#SBATCH --output=slurm/finetune_chartqa_%j.out
#SBATCH --error=slurm/finetune_chartqa_%j.err
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
# HuggingFace Cache variables
export HF_HOME="/data/user_data/bsood/.hf_cache"
export HF_HUB_CACHE="/data/user_data/bsood/.hf_cache/hub"
export HF_DATASETS_CACHE="/data/user_data/bsood/.hf_cache/datasets"

# Path to your pre-trained projector weights
PROJECTOR_WEIGHTS="/home/bsood/Multimodal-Machine-Learning-Project/outputs/v_large_lr_1_gpu/checkpoint-10465/mihika_projector_1e4.pt"
EXPERIMENT_NAME="chartqa_finetune_mihika_projector_1e4"

echo "Starting ChartQA Fine-Tuning"
echo "Using Projector Weights: $PROJECTOR_WEIGHTS"
nvidia-smi

python3 finetune_chartqa.py \
    --experiment_name $EXPERIMENT_NAME \
    --projector_weights $PROJECTOR_WEIGHTS \
    --epochs 3 \
    --lr 7e-5

echo "Job finished."