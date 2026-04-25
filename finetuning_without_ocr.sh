#!/bin/bash
#SBATCH --job-name=finetuning_without_ocr
#SBATCH --partition=general
#SBATCH --array=1-4
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --cpus-per-task=6
#SBATCH --time=6:00:00
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --constrain=A100_80GB

case "$SLURM_ARRAY_TASK_ID" in
  1)
    LOG_NAME="overfit"
    CMD=(python3 finetuning_without_ocr.py --experiment_name overfit --projector_weights outputs/7e3_a80_mediumishish_batch.pt --run_ocr_eval --overfit_check --epochs 100)
    ;;
  2) 
    LOG_NAME="llava"
    CMD=(python3 finetuning_without_ocr.py --experiment_name llava --projector_weights outputs/7e3_a80_mediumishish_batch.pt --run_ocr_eval)
    ;;
  3) 
    LOG_NAME="summarization"
    CMD=(python3 finetuning_without_ocr.py --experiment_name summarization --projector_weights outputs/summarization_7e3.pt --run_ocr_eval --eval_only --ckpt_dir outputs/summarization/epoch_artifacts/epoch_2)
    ;;
  4) 
    LOG_NAME="table"
    CMD=(python3 finetuning_without_ocr.py --experiment_name table --projector_weights outputs/table_7e3.pt --run_ocr_eval)
    ;;
  5) 
    LOG_NAME="llava_small"
    CMD=(python3 finetuning_without_ocr.py --experiment_name llava_small --projector_weights outputs/7e3_a80_mediumishish_batch.pt --run_ocr_eval --per_device_train_batch_size 8 --eval_only --ckpt_dir outputs/llava_small/epoch_artifacts/epoch_2)
    ;;
  6) 
    LOG_NAME="summarization_small"
    CMD=(python3 finetuning_without_ocr.py --experiment_name summarization_small --projector_weights outputs/summarization_7e3.pt --run_ocr_eval --per_device_train_batch_size 8)
    ;;
  7) 
    LOG_NAME="table_small"
    CMD=(python3 finetuning_without_ocr.py --experiment_name table_small --projector_weights outputs/table_7e3.pt --run_ocr_eval --per_device_train_batch_size 8 --eval_only --ckpt_dir outputs/table_small/epoch_artifacts/epoch_2)
    ;;
  8) 
    LOG_NAME="summarization_large_lr"
    CMD=(python3 finetuning_without_ocr.py --experiment_name summarization_large_lr --projector_weights outputs/summarization_7e3.pt --run_ocr_eval --per_device_train_batch_size 8 --lr 1e-3)
    ;;
  9) 
    LOG_NAME="chart"
    CMD=(python3 finetuning_without_ocr.py --experiment_name chart --projector_weights outputs/chartqa_1e4.pt --run_ocr_eval --eval_only --ckpt_dir outputs/chart/epoch_artifacts/epoch_2)
    ;;
  10) 
    LOG_NAME="table_small_lr"
    CMD=(python3 finetuning_without_ocr.py --experiment_name table_small_lr --projector_weights outputs/table_7e3.pt --run_ocr_eval --per_device_train_batch_size 8 --lora_r 16 --lora_alpha 32 --eval_only --ckpt_dir outputs/table_small_lr/epoch_artifacts/epoch_2)
    ;;
  11) 
    LOG_NAME="llava_small_1e_1e4"
    CMD=(python3 finetuning_without_ocr.py --experiment_name llava_small_1e_1e4 --projector_weights outputs/1e4_a80_mediumishish_batch_1e/checkpoint-5233/projector.pt --run_ocr_eval --per_device_train_batch_size 8)
    ;;
  12) 
    LOG_NAME="llava_small_1e_1e3"
    CMD=(python3 finetuning_without_ocr.py --experiment_name llava_small_1e_1e3 --projector_weights outputs/1e3_a80_mediumishish_batch_1e/checkpoint-5233/projector.pt --run_ocr_eval --per_device_train_batch_size 8)
    ;;
  13) 
    LOG_NAME="summarization_small_1e_1e3"
    CMD=(python3 finetuning_without_ocr.py --experiment_name summarization_small_1e_1e3 --projector_weights outputs/summarization_1e3_1e4_1e.pt --run_ocr_eval --per_device_train_batch_size 8)
    ;;
  14) 
    LOG_NAME="summarization_small_1e_1e4"
    CMD=(python3 finetuning_without_ocr.py --experiment_name summarization_small_1e_1e4 --projector_weights outputs/summarization_1e4_1e4_1e/checkpoint-2696/projector.pt --run_ocr_eval --per_device_train_batch_size 8)
    ;;
  15) 
    LOG_NAME="summarization_small_actual_1e"
    CMD=(python3 finetuning_without_ocr.py --experiment_name summarization_small_actual_1e --projector_weights outputs/summarization_1e4_actual_1e/checkpoint-1348/projector.pt --run_ocr_eval --per_device_train_batch_size 8)
    ;;
  16) 
    LOG_NAME="table_small_1e"
    CMD=(python3 finetuning_without_ocr.py --experiment_name table_small_1e --projector_weights outputs/table_1e4_1e/checkpoint-56/projector.pt --run_ocr_eval --per_device_train_batch_size 8)
    ;;
  17) 
    LOG_NAME="table_small_2e"
    CMD=(python3 finetuning_without_ocr.py --experiment_name table_small_2e --projector_weights outputs/table_1e4_2e/checkpoint-112/projector.pt --run_ocr_eval --per_device_train_batch_size 8)
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
