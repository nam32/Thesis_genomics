#!/bin/bash
#SBATCH --job-name=featureCounts_all
#SBATCH --output=3_featureCounts_all_%j.out
#SBATCH --error=3_featureCounts_all_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load Subread/2.0.3-GCC-10.3.0

# Base directory containing all the sample folders
BASE_DIR="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/mabin/star_bams_with_counts_50multimap"
output_path="/storage/homefs/tj23t050/snRNA_variant_predictions/results/3_RNA-seq_count"

mkdir -p /storage/homefs/tj23t050/snRNA_variant_predictions/results/3_RNA-seq_count

# Collect all BAM files across all subdirectories
BAM_FILES=$(find "$BASE_DIR" -name "*_Aligned.sortedByCoord.out.bam" | sort)

# Run featureCounts on all BAMs
featureCounts \
  -a /storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/Nam/data/Ensembl/gencode.v47.annotation_rmchr.gtf \
  -o $output_path/star_bams_with_counts_50multimap_counts_gene_name.txt \
  -T 8 \
  -g gene_name \
  -O --fraction \
  -p \
  $BAM_FILES
