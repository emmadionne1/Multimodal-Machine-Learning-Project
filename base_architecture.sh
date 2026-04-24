#!/bin/bash
#SBATCH --job-name=base_architecture
#SBATCH --partition=general
#SBATCH --array=1-14
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --constrain=A100_40GB|L40S

case "$SLURM_ARRAY_TASK_ID" in
  # OVERFIT CHECKS
  1)
    LOG_NAME="overfit_check"
    CMD=(python3 base_architecture.py --experiment_name overfit_check --overfit_check --epochs 100 --lr 5e-5 --task pretrain)
    ;;
  2)
    LOG_NAME="overfit_check_chartqa"
    CMD=(python3 base_architecture.py --experiment_name overfit_check_chartqa --overfit_check --epochs 100 --lr 5e-5 --task chartqa)
    ;;
  # CHARTQA RUN
  3)
    LOG_NAME="chartqa_llava_direct"
    CMD=(python3 base_architecture.py --experiment_name chartqa_llava_direct --epochs 5 --lr 5e-3 --task chartqa --trained_weights outputs/1e4_a80_mediumishish_batch_1e/checkpoint-5233/projector.pt --per_device_train_batch_size 8)
    ;;
  # BATCH SIZE 12 RUNS ACROSS LRS
  4) 
    LOG_NAME="7e3_a40"
    CMD=(python3 base_architecture.py --experiment_name 7e3_a40 --lr 7e-3 --epochs 2 --task pretrain --per_device_train_batch_size 12)
    ;;
  5) 
    LOG_NAME="1e4_a40"
    CMD=(python3 base_architecture.py --experiment_name 1e4_a40 --lr 1e-4 --epochs 2 --task pretrain --per_device_train_batch_size 12)
    ;;
  6) 
    LOG_NAME="1e3_a40"
    CMD=(python3 base_architecture.py --experiment_name 1e3_a40 --lr 1e-3 --epochs 2 --task pretrain --per_device_train_batch_size 12)
    ;;
  7) 
    LOG_NAME="3e4_a40"
    CMD=(python3 base_architecture.py --experiment_name 3e4_a40 --lr 3e-4 --epochs 2 --task pretrain --per_device_train_batch_size 12)
    ;;
  # BATCH SIZE 16 RUNS ACROSS LRS
  10) 
    LOG_NAME="1e3_a80"
    # sbatch --array=10 --constrain=A100_80GB base_architecture.sh
    CMD=(python3 base_architecture.py --experiment_name 1e3_a80 --lr 1e-3 --epochs 2 --task pretrain)
    ;;
  11) 
    LOG_NAME="7e3_a80"
    # sbatch --array=11 --constrain=A100_80GB base_architecture.sh
    CMD=(python3 base_architecture.py --experiment_name 7e3_a80 --lr 7e-3 --epochs 2 --task pretrain)
    ;;
  # BATCH SIZE 24 RUNS
  13)
    LOG_NAME="1e3_a80_mediumishish_batch"
    # sbatch --array=13 --constrain=A100_80GB base_architecture.sh
    CMD=(python3 base_architecture.py --experiment_name 1e3_a80_mediumishish_batch --lr 1e-3 --epochs 2 --task pretrain --per_device_train_batch_size 24)
    ;;
  14)
    LOG_NAME="1e3_a80_medium_batch"
    # sbatch --array=14 --gpus=2 --constrain=A100_80GB base_architecture.sh
    CMD=(python3 base_architecture.py --experiment_name 1e3_a80_medium_batch --lr 1e-3 --epochs 2 --task pretrain --per_device_train_batch_size 24 --gradient_accumulation_steps 8)
    ;;
  8)
    LOG_NAME="7e3_a80_mediumishish_batch"
    # sbatch --array=8 --constrain=A100_80GB base_architecture.sh
    CMD=(python3 base_architecture.py --experiment_name 7e3_a80_mediumishish_batch --lr 7e-3 --epochs 2 --task pretrainn --per_device_train_batch_size 24)
    ;;
  21)
    LOG_NAME="1e3_a80_mediumishish_batch_1e"
    # sbatch --array=21 --constrain=A100_80GB base_architecture.sh
    CMD=(python3 base_architecture.py --experiment_name 1e3_a80_mediumishish_batch_1e --lr 1e-3 --epochs 1 --task pretrain --per_device_train_batch_size 24)
    ;;
  23)
    LOG_NAME="1e4_a80_mediumishish_batch_1e"
    # sbatch --array=23 --constrain=A100_80GB base_architecture.sh
    CMD=(python3 base_architecture.py --experiment_name 1e4_a80_mediumishish_batch_1e --lr 1e-4 --epochs 1 --task pretrain --per_device_train_batch_size 24)
    ;;
  # CHART SUMMARIZATION
  12)
    # sbatch --array=12 --constrain=A100_80GB base_architecture.sh
    LOG_NAME="summarization_1e5"
    CMD=(python3 base_architecture.py --experiment_name summarization_1e5 --lr 1e-5 --task summarization 
          --trained_weights outputs/7e3_a80_mediumishish_batch.pt --per_device_train_batch_size 8 --gradient_accumulation_steps 2)
    ;;
  9)
    # sbatch --array=9 --constrain=A100_80GB base_architecture.sh
    LOG_NAME="summarization_1e4"
    CMD=(python3 base_architecture.py --experiment_name summarization_1e4 --lr 1e-4 --task summarization 
          --trained_weights outputs/7e3_a80_mediumishish_batch.pt --per_device_train_batch_size 8 --gradient_accumulation_steps 2)
    ;;
  16)
    # sbatch --array=16 --constrain=A100_80GB base_architecture.sh
    LOG_NAME="summarization_1e3"
    CMD=(python3 base_architecture.py --experiment_name summarization_1e3 --lr 1e-3 --task summarization 
          --trained_weights outputs/7e3_a80_mediumishish_batch.pt --per_device_train_batch_size 8 --gradient_accumulation_steps 2)
    ;;
  17)
    # sbatch --array=17 --constrain=A100_80GB base_architecture.sh
    LOG_NAME="summarization_7e3"
    CMD=(python3 base_architecture.py --experiment_name summarization_7e3 --lr 7e-3 --task summarization 
          --trained_weights outputs/7e3_a80_mediumishish_batch.pt --per_device_train_batch_size 8 --gradient_accumulation_steps 2)
    ;;
  22)
    # sbatch --array=22 --constrain=A100_80GB base_architecture.sh
    LOG_NAME="summarization_1e4_1e4_1e"
    CMD=(python3 base_architecture.py --experiment_name summarization_1e4_1e4_1e --lr 1e-4 --task summarization 
          --trained_weights outputs/1e4_a80_mediumishish_batch_1e/checkpoint-5233/projector.pt --per_device_train_batch_size 8 --gradient_accumulation_steps 2)
    ;;
  24)
    # sbatch --array=24 --constrain=A100_80GB base_architecture.sh
    LOG_NAME="summarization_1e3_1e4_1e"
    CMD=(python3 base_architecture.py --experiment_name summarization_1e3_1e4_1e --lr 1e-3 --task summarization 
          --trained_weights outputs/1e4_a80_mediumishish_batch_1e/checkpoint-5233/projector.pt --per_device_train_batch_size 8 --gradient_accumulation_steps 2)
    ;;
  # CHART TO TABLE
  15)
    # sbatch --array=15 --constrain=A100_80GB --time=2:00:00 base_architecture.sh
    LOG_NAME="table_1e4"
    CMD=(python3 base_architecture.py --experiment_name table_1e4 --lr 1e-4 --epochs 3 --task table --max_new_tokens 1000
          --trained_weights outputs/summarization_7e3.pt --per_device_train_batch_size 4 --gradient_accumulation_steps 4)
    ;;
  18)
    # sbatch --array=18 --constrain=A100_80GB --time=2:00:00 base_architecture.sh
    LOG_NAME="table_1e3"
    CMD=(python3 base_architecture.py --experiment_name table_1e3 --lr 1e-3 --epochs 3 --task table --max_new_tokens 1000
          --trained_weights outputs/summarization_7e3.pt --per_device_train_batch_size 4 --gradient_accumulation_steps 4)
    ;;
  19)
    # sbatch --array=19 --constrain=A100_80GB --time=2:00:00 base_architecture.sh
    LOG_NAME="table_7e3"
    CMD=(python3 base_architecture.py --experiment_name table_7e3 --lr 7e-3 --epochs 3 --task table --max_new_tokens 1000
          --trained_weights outputs/summarization_7e3.pt --per_device_train_batch_size 4 --gradient_accumulation_steps 4)
    ;;
  # CHARTQA
  20)
    # sbatch --array=20 --constrain=A100_80GB --time=5:00:00 base_architecture.sh
    LOG_NAME="chartqa_1e4"
    CMD=(python3 base_architecture.py --experiment_name chartqa_1e4 --epochs 3 --lr 1e-4 --task chartqa --trained_weights outputs/table_7e3.pt)
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

export HF_HOME="/data/user_data/$USER/.hf_cache"
export HF_HUB_CACHE="/data/hf_cache/hub"
export HF_DATASETS_CACHE="/data/hf_cache/datasets"

"${CMD[@]}"
