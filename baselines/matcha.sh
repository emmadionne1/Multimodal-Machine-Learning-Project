#!/bin/bash
#SBATCH --job-name=matcha_baseline
#SBATCH --partition=preempt
#SBATCH --output=slurm/matcha_baseline.out
#SBATCH --error=slurm/matcha_baseline.err
#SBATCH --export=ALL
#SBATCH --constrain=L40|L40S
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gpus=2

source $HOME/.bashrc
conda init
conda activate mmml
cd $HOME/Multimodal-Machine-Learning-Project/baselines

# Reset the log files
> slurm/matcha_baseline.out
> slurm/matcha_baseline.err

nvidia-smi
python3 multimodal_prompting_baselines.py --experiment_type zero --model_name google/matcha-chartqa