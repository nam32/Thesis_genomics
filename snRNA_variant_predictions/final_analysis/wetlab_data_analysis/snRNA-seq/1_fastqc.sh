#!/usr/bin/env bash

#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=run_blastdb
#SBATCH --partition=epyc2
#SBATCH --output=fastqc_%j.out
#SBATCH --error=fastqc_%j.err

module load FastQC/0.11.9-Java-11

# Define variables for clarity
INPUT_DIR="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/Kaitlin_data/snRNA_seq/merged_fastq"
OUTPUT_DIR="/storage/homefs/tj23t050/snRNA_variant_predictions/results/rubin_lab_snRNA_fastqc"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run FastQC with specified input and output directories
fastqc -t 4 -o "$OUTPUT_DIR" "$INPUT_DIR"/*.fastq.gz
