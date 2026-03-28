#!/bin/bash
#SBATCH --job-name=base_architecture
#SBATCH --partition=general
#SBATCH --array=1-2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --ntasks=1

case "$SLURM_ARRAY_TASK_ID" in
  1)
    LOG_NAME="overfit_check"
    CMD=(python3 base_architecture.py --experiment_name overfit_check --overfit_check --epochs 100 --lr 5e-5 --pretrain)
    ;;
  2)
    LOG_NAME="v_large_lr"
    CMD=(python3 base_architecture.py --experiment_name v_large_lr_1_gpu --lr 1e-4 --epochs 1 --pretrain)
    ;;
  3)
    LOG_NAME="v_large_lr_backup"
    CMD=(python3 base_architecture.py --experiment_name v_large_lr_backup --pretrain)
    ;;
  4)
    LOG_NAME="v_large_lr_backup_tiny"
    CMD=(python3 base_architecture.py --experiment_name v_large_lr_backup_tiny --dataset_split_index 0 --pretrain)
    ;;
  5) 
    LOG_NAME="1e4_a40"
    # sbatch --array=5 --gpus=1 --constrain=A100_40GB --mem=80G --cpus-per-task=6 --time=12:00:00 --partition array base_architecture.sh
    CMD=(python3 base_architecture.py --experiment_name 1e4_a40 --lr 1e-4 --epochs 2 --pretrain)
    ;;
  6) 
    LOG_NAME="1e5_a40"
    # sbatch --array=6 --gpus=1 --constrain=A100_40GB --mem=80G --cpus-per-task=6 --time=6:00:00 --partition array base_architecture.sh
    CMD=(python3 base_architecture.py --experiment_name 1e5_a40 --lr 1e-5 --epochs 2 --pretrain)
    ;;
  7) 
    LOG_NAME="3e4_a40"
    # sbatch --array=7 --gpus=1 --constrain=A100_40GB --mem=80G --cpus-per-task=6 --time=6:00:00 --partition array base_architecture.sh
    CMD=(python3 base_architecture.py --experiment_name 3e4_a40 --lr 3e-4 --epochs 2 --pretrain)
    ;;
  8)
    LOG_NAME="overfit_check_chartqa"
    CMD=(python3 base_architecture.py --experiment_name overfit_check_chartqa --overfit_check --epochs 100 --lr 5e-5)
    ;;
  9)
    LOG_NAME="chartqa_direct"
    # sbatch --array=9 --gpus=2 --constrain=A100_40GB --mem=80G --cpus-per-task=6 --time=6:00:00 --partition array base_architecture.sh
    CMD=(python3 base_architecture.py --experiment_name chartqa_direct --epochs 5 --lr 5e-4)
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

OUT_FILE="slurm/${LOG_NAME}4.out"
ERR_FILE="slurm/${LOG_NAME}4.err"

mkdir -p slurm

# Reset the log files
> "$OUT_FILE"
> "$ERR_FILE"

# Redirect everything from here onward
exec >"$OUT_FILE" 2>"$ERR_FILE"

source "$HOME/.bashrc"
conda init
conda activate bsampling
cd "$HOME/Multimodal-Machine-Learning-Project"

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "LOG_NAME=$LOG_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=pci.bus_id
nvidia-smi -q -d PAGE_RETIREMENT
nvidia-smi

export HF_HOME="/data/user_data/bsood/.hf_cache"
export HF_HUB_CACHE="/data/hf_cache/hub"
export HF_DATASETS_CACHE="/data/hf_cache/datasets"

"${CMD[@]}"