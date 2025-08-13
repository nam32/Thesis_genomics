#!/bin/bash
#SBATCH --job-name=dnabert2-predict_binary
#SBATCH --output=logs/dnabert2-predict_binary-%j.out
#SBATCH --error=logs/dnabert2-predict_binary-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu
#SBATCH --gpus-per-node=rtx3090:1

# DNABERT-2 Prediction Job
# This SLURM script runs the `predict_binary.py` script using DNABERT-2
# in a conda environment (`dnabert2_env`) with GPU acceleration.
# Adjust GPU type and memory requirements according to your cluster's hardware.

# dnabert2_env

module load cuda/11.8

# Activate conda
source ~/.bashrc
conda activate dnabert2_env

# Make sure logs dir exists
mkdir -p logs

# Run your training script
python predict_binary.py
