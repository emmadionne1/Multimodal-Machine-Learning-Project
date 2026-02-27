#!/bin/bash
#SBATCH --job-name=text
#SBATCH --partition=preempt
#SBATCH --output=slurm/text.out
#SBATCH --error=slurm/text.err
#SBATCH --export=ALL
#SBATCH --constrain=L40|L40S
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=48G
#SBATCH --gpus=2

source $HOME/.bashrc
conda init
conda activate mmml
cd $HOME/Multimodal-Machine-Learning-Project/baselines

# Reset the log files
> slurm/text.out
> slurm/text.err

nvidia-smi
python3 text_only_baseline.py --experiment_type zero