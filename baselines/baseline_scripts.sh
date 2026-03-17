#!/bin/bash
#SBATCH --job-name=baseline_scripts
#SBATCH --partition=preempt
#SBATCH --array=1-7
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --constrain=L40|L40S
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gpus=2

case "$SLURM_ARRAY_TASK_ID" in
  1)
    LOG_NAME="both_new_model"
    CMD=(python3 multimodal_prompting_baselines.py --experiment_type zero --model_name Qwen/Qwen3-VL-4B-Instruct)
    ;;
  2)
    LOG_NAME="both"
    CMD=(python3 multimodal_prompting_baselines.py --experiment_type zero)
    ;;
  3)
    LOG_NAME="matcha_baseline"
    CMD=(python3 multimodal_prompting_baselines.py --experiment_type zero --model_name google/matcha-chartqa)
    ;;
  4)
    LOG_NAME="mlp_baseline"
    CMD=(python3 mlp_baseline2.py)
    ;;
  5)
    LOG_NAME="text_few"
    CMD=(python3 text_only_baseline.py --experiment_type few)
    ;;
  6)
    LOG_NAME="text_new_model"
    CMD=(python3 text_only_baseline.py --experiment_type zero --model_name Qwen/Qwen2.5-3B-Instruct)
    ;;
  7)
    LOG_NAME="text"
    CMD=(python3 text_only_baseline.py --experiment_type zero)
    ;;
  8)
    LOG_NAME="both_few"
    CMD=(python3 multimodal_prompting_baselines.py --experiment_type few)
    ;;
  9)
    LOG_NAME="both_few_new_model"
    CMD=(python3 multimodal_prompting_baselines.py --experiment_type few --model_name Qwen/Qwen3-VL-4B-Instruct)
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
cd "$HOME/Multimodal-Machine-Learning-Project/baselines" || exit 1

export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "LOG_NAME=$LOG_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi

"${CMD[@]}"