#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx
#SBATCH --mem-per-gpu 40G
#SBATCH -o slurm-logs/slurm-%x-%j-%N.out

# env | grep CUDA
# nvidia-smi -L

./run.sh "$@"
