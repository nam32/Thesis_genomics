#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=plot_snRNA_chip_summary
#SBATCH --output=1_plot_%j.out
#SBATCH --error=1_plot_%j.err

python plot_final.py