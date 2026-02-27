#!/bin/bash
#SBATCH --job-name=gemma_baseline
#SBATCH --partition=preempt
#SBATCH --output=slurm/gemma_baseline.out
#SBATCH --error=slurm/gemma_baseline.err
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
> slurm/gemma_baseline.out
> slurm/gemma_baseline.err

nvidia-smi
python3 multimodal_prompting_baselines.py --experiment_type zero --model_name ahmed-masry/chartgemma