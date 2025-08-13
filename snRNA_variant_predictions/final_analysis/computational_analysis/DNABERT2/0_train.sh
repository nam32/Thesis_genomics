#!/bin/bash
#SBATCH --job-name=dnabert2-train
#SBATCH --output=logs/dnabert2-train-%j.out
#SBATCH --error=logs/dnabert2-train-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu
#SBATCH --gpus-per-node=rtx3090:1

# dnabert2_env

# module load cuda/11.8

# Activate conda
source ~/.bashrc
conda activate dnabert2_env

# Make sure logs dir exists
mkdir -p logs

# Run your training script
python train_model.py
