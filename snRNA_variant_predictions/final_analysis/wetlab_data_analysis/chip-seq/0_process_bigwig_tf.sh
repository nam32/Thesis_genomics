#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=snRNA_chipseq_summary
#SBATCH --output=convert_pol_chipseq_snRNA_tf_%j.out
#SBATCH --error=convert_pol_chipseq_snRNA_tf_%j.err

# Load environment
source activate deeptools_env

# Set directories and input
chipseq_list="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/chip-seq_tf/chipseq_bigwigs.txt"
output_dir="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/chip-seq_analysis_tf_bigwig"
snRNA="/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results/rfam_ensembl_infernal/final_snRNA_combined.TSSm70_m10.bed"

mkdir -p "$output_dir"
cd "$output_dir"

# Loop through BigWig paths from list
while IFS= read -r bigwig_path; do

    if [[ ! -f "$bigwig_path" ]]; then
        echo "⚠️  BigWig file not found: $bigwig_path"
        continue
    fi

    # Extract accession or file name for naming output
    accession=$(basename "$bigwig_path" .bigWig)
    output_prefix="${accession}_snRNA"

    echo "✅ Processing $accession..."

    multiBigwigSummary BED-file \
      --bwfiles "$bigwig_path" \
      --BED "$snRNA" \
      --outRawCounts "${output_prefix}_signal_matrix.tab" \
      --outFileName "${output_prefix}_summary.npz" \
      --numberOfProcessors $SLURM_CPUS_PER_TASK

done < "$chipseq_list"
