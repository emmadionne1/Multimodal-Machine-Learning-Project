#!/bin/bash
#SBATCH --job-name=mlp_baseline
#SBATCH --partition=preempt
#SBATCH --output=slurm/mlp_baseline.out
#SBATCH --error=slurm/mlp_baseline.err
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
> slurm/mlp_baseline.out
> slurm/mlp_baseline.err

nvidia-smi
python3 mlp_baseline.py