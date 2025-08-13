#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=convert_bigwig
#SBATCH --output=convert_bigwig_%j.out
#SBATCH --error=convert_bigwig_%j.err

conda activate deeptools_env

# Set directories
summary_file="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/chip-seq/files_summary.tsv"
data_dir="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/chip-seq"
output_dir="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/chip-seq_analysis"

mkdir -p "$output_dir"

# Begin per-experiment processing
cd "$output_dir"

# Loop through each line in your metadata-matching file
while IFS=$'\t' read -r bigwig bed bigbed accession; do
    echo "Processing $accession..."

    bigwig_path="$data_dir/$bigwig"
    bed_path="$data_dir/$bed"
    output_prefix="${accession}_snRNA"

    multiBigwigSummary BED-file \
      --bwfiles "$bigwig_path" \
      --BED "$bed_path" \
      --outRawCounts "${output_prefix}_signal_matrix.tab" \
      --outFileName "${output_prefix}_summary.npz" \
      --numberOfProcessors $SLURM_CPUS_PER_TASK

done < $summary_file