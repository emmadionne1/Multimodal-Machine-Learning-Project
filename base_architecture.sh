#!/bin/bash
#SBATCH --job-name=base_architecture
#SBATCH --partition=general
#SBATCH --array=1-2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --cpus-per-task=6
#SBATCH --time=30:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:A100_40GB:1
#SBATCH --ntasks=1

case "$SLURM_ARRAY_TASK_ID" in
  1)
    LOG_NAME="overfit_check"
    CMD=(python3 base_architecture.py --experiment_name overfit_check --overfit_check --epochs 100)
    ;;
  2)
    LOG_NAME="trial_run_full_datasetA"
    CMD=(python3 base_architecture.py --experiment_name final_run_full_dataset_A40)
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

OUT_FILE="slurm/${LOG_NAME}_4.out"
ERR_FILE="slurm/${LOG_NAME}_4.err"

mkdir -p slurm

# Reset the log files
> "$OUT_FILE"
> "$ERR_FILE"

# Redirect everything from here onward
exec >"$OUT_FILE" 2>"$ERR_FILE"

source ~/miniconda3/bin/activate
conda activate bsampling
cd "$HOME/Multimodal-Machine-Learning-Project"

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "LOG_NAME=$LOG_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=pci.bus_id
nvidia-smi -q -d PAGE_RETIREMENT
nvidia-smi
# python3 -m pip uninstall -y torch torchvision torchaudio
# python3 -m pip install --no-cache-dir \
#   torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
#   --index-url https://download.pytorch.org/whl/cu128


python3 -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count()); print('bf16 supported', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'no cuda')"
export HF_HOME="/data/user_data/bsood/.hf_cache"
export HF_HUB_CACHE="/data/hf_cache/hub"
export HF_DATASETS_CACHE="/data/hf_cache/datasets"

"${CMD[@]}"