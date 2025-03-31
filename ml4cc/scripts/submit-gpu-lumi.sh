#!/bin/bash
#SBATCH --job-name=ml4cc
#SBATCH --account=project_465001293
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gpus-per-task=1
#SBATCH --partition=small-g
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

./run-lumi.sh "$@"
