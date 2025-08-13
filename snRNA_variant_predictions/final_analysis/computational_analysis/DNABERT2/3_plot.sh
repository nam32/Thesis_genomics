#!/bin/bash
#SBATCH --job-name=plot_val
#SBATCH --output=logs/plot-%j.out
#SBATCH --error=logs/plot-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# dnabert2_env

module load cuda/11.8

# Activate conda
source ~/.bashrc
conda activate dnabert2_env

# Make sure logs dir exists
mkdir -p logs

# Run your training script
python plot.py
