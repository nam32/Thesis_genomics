#!/bin/bash
#
# SLURM job: Run MEME Suite FIMO on one or more FASTA files with JASPAR motifs.
# - Scans both strands by default.
# - Writes standard FIMO outputs into a per-FASTA folder under $OUTPUT_BASE.
# - Use THRESH (p-value) or QV_THRESH (q-value) to control stringency.
#
# Tips:
# - If your FASTA headers include genomic coords (e.g., chr:start-end(+/-)),
#   uncomment --parse-genomic-coord for nicer coordinates in results.
# - Provide a background model with --bgfile if you have genome-specific bg.
#
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=fimo_run
#SBATCH --output=fimo_run_%j.out
#SBATCH --error=fimo_run_%j.err

# Path to motif matrix in MEME format (JASPAR)
meme_pfms_path="/storage/homefs/tj23t050/snRNA_variant_predictions/scripts/taxonomy_paper/JASPAR2024_CORE_non-redundant_pfms_meme.txt"

# Input FASTA files (explicitly listed)
fasta_files=(
"/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results/snRNA_gene_gencode_combined_sorted_upstream_1000_fimo_ver.fa"
)

# Output base directory
output_base="/storage/homefs/tj23t050/snRNA_variant_predictions/results/1_fimo"
mkdir -p "$output_base"

# Loop through each FASTA file
for fasta in "${fasta_files[@]}"; do
    base=$(basename "$fasta" .fa)
    output_dir="${output_base}/fimo_${base}"
    mkdir -p "$output_dir"

    echo "Running FIMO on $base → $output_dir"
    fimo --oc "$output_dir" \
         --max-stored-scores 1000000 \
         "$meme_pfms_path" "$fasta"
done
