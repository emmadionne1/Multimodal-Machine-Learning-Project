#!/bin/bash
#SBATCH --job-name=ocr_eval_chartqa_lora
#SBATCH --partition=general
#SBATCH --output=slurm/ocr_eval_chartqa_lora_%j.out
#SBATCH --error=slurm/ocr_eval_chartqa_lora_%j.err
#SBATCH --export=ALL
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --ntasks=1

mkdir -p slurm

source "$HOME/.bashrc"
conda activate bsampling
cd "$HOME/Multimodal-Machine-Learning-Project"

export HF_HOME="/data/user_data/bsood/.hf_cache"
export HF_HUB_CACHE="/data/user_data/bsood/.hf_cache/hub"
export HF_DATASETS_CACHE="/data/user_data/bsood/.hf_cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Point this to the saved LoRA checkpoint directory, not projector.pt
# Example final checkpoint:
CKPT_DIR="/home/bsood/Multimodal-Machine-Learning-Project/outputs/chartqa_finetune_bhavesh_pretrained_lora/chartqa_final"

# Example intermediate checkpoint:
# CKPT_DIR="/home/bsood/Multimodal-Machine-Learning-Project/outputs/chartqa_finetune_bhavesh_pretrained_lora/stage2_checkpoints/stage2_step_360"

echo "=================================================="
echo "Starting ChartQA OCR LoRA Evaluation"
echo "Checkpoint dir: $CKPT_DIR"
echo "=================================================="



nvidia-smi

python evaluate_with_ocr.py \
    --ckpt_dir "$CKPT_DIR" \
    --split test \
    --max_new_tokens 45

echo "Job finished."