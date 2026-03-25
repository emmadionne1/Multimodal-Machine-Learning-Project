#!/bin/bash
#SBATCH --job-name=base_architecture
#SBATCH --partition=general
#SBATCH --array=1-2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --cpus-per-task=2
#SBATCH --time=30:00:00
#SBATCH --mem=48G
#SBATCH --gpus=3
#SBATCH --ntasks=1
#SBATCH --constrain=A100_80GB|L40S|6000Ada

case "$SLURM_ARRAY_TASK_ID" in
  1)
    LOG_NAME="overfit_check"
    CMD=(python3 base_architecture.py --experiment_name overfit_check --overfit_check --epochs 100 --lr 5e-5)
    ;;
  2)
    LOG_NAME="v_large_lr"
    CMD=(python3 base_architecture.py --experiment_name v_large_lr)
    ;;
  3)
    LOG_NAME="v_large_lr_tiny_dataset"
    CMD=(python3 base_architecture.py --experiment_name v_large_lr_tiny_dataset --dataset_split_index 0)
    ;;
  4)
    LOG_NAME="v_large_lr_tiny_dataset_preempt"
    CMD=(python3 base_architecture.py --experiment_name v_large_lr_tiny_dataset_preempt --dataset_split_index 0)
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

OUT_FILE="slurm/${LOG_NAME}.out"
ERR_FILE="slurm/${LOG_NAME}.err"

mkdir -p slurm

# Reset the log files
> "$OUT_FILE"
> "$ERR_FILE"

# Redirect everything from here onward
exec >"$OUT_FILE" 2>"$ERR_FILE"

source "$HOME/.bashrc"
conda init
conda activate mmml
cd "$HOME/Multimodal-Machine-Learning-Project"

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "LOG_NAME=$LOG_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=pci.bus_id
nvidia-smi -q -d PAGE_RETIREMENT
nvidia-smi

export HF_HOME="/data/user_data/mbairath/.hf_cache"
export HF_HUB_CACHE="/data/hf_cache/hub"
export HF_DATASETS_CACHE="/data/hf_cache/datasets"

srun "${CMD[@]}"