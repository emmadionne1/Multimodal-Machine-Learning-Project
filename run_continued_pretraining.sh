#!/bin/bash
#SBATCH --job-name=a6finetune_chartqa
#SBATCH --partition=general
#SBATCH --output=slurm/cont/cont_pretrain_finetune_chartqa_%j.out
#SBATCH --error=slurm/cont/cont_pretrain_finetune_chartqa_%j.err
#SBATCH --export=ALL
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:6000Ada:2
#SBATCH --ntasks=1

mkdir -p slurm

source "$HOME/.bashrc"
conda activate bsampling
cd "$HOME/Multimodal-Machine-Learning-Project"

export HF_HOME="/data/user_data/bsood/.hf_cache"
export HF_HUB_CACHE="/data/user_data/bsood/.hf_cache/hub"
export HF_DATASETS_CACHE="/data/user_data/bsood/.hf_cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECTOR_WEIGHTS="/home/bsood/Multimodal-Machine-Learning-Project/outputs/bhavesh-epoch3 checkpoint-1770/projector.pt"
EXPERIMENT_NAME="chartqa_finetune_bhavesh_pretrained_lora"

echo "Using Projector Weights: $PROJECTOR_WEIGHTS"

python -m pip uninstall -y deepspeed
nvidia-smi

torchrun --standalone --nproc_per_node=2 continued_pretraining_and_finetuning.py \
  --experiment_name "$EXPERIMENT_NAME" \
  --skip_stage1 \
  --resume_peft_dir "/home/bsood/Multimodal-Machine-Learning-Project/outputs/chartqa_finetune_bhavesh_pretrained_lora/stage1_checkpoints/stage1_step_5233"\
  --epochs 3 \
  --lr 1e-4 \
  --finetune_batch_size 4 \
  --finetune_grad_accum 8 \
  --eval_batch_size 4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --saves_per_epoch 3 \
  --max_ocr_words 1000