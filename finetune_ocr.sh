#!/bin/bash
#SBATCH --job-name=ocr_finetune_chartqa
#SBATCH --partition=general
#SBATCH --output=slurm/ocr_finetune_chartqa_%j.out
#SBATCH --error=slurm/ocr_finetune_chartqa_%j.err
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

# Path to the specific checkpoint you requested
PROJECTOR_WEIGHTS="/home/bsood/Multimodal-Machine-Learning-Project/outputs/chartqa_finetune_mihika_projector_1e4/projector.pt"
EXPERIMENT_NAME="chartqa_ocr_finetune_mihika_projector_1e4"

echo "=================================================="
echo "Starting ChartQA EasyOCR Pipeline"
echo "Checkpoint: $PROJECTOR_WEIGHTS"
echo "=================================================="
nvidia-smi

# Note: --epochs 1 ensures we only train for a single epoch as requested.
python3 finetune_ocr.py \
    --experiment_name $EXPERIMENT_NAME \
    --projector_weights "$PROJECTOR_WEIGHTS" \
    --epochs 1 \
    --lr 5e-5

echo "Job finished."