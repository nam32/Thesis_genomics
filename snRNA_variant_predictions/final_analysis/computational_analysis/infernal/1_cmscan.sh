#!/usr/bin/env bash
#
# SLURM array job: run Infernal `cmscan` across a list of covariance models (.cm)
# One array task = one CM file from CM_LIST (0-based indexing).
#
# Outputs:
#   results_dir/<snrna>.tblout   # tabular hits
#   (default stdout goes to SLURM logs)
#
# Notes:
# - `--cut_ga` uses model-specific GA thresholds (recommended if curated).
# - `--noali` saves time/space by omitting alignment in the human-readable output.
# - Make sure `cmscan` is available (module or conda env).
#
# Optional:
# - After all tasks finish, you can concat *.tblout into one file.
# - If your CM list includes comments or blank lines, they’re ignored below.

#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=cmscan_array
#SBATCH --partition=epyc2
#SBATCH --output=cmscan_%A_%a.out
#SBATCH --error=cmscan_%A_%a.err
#SBATCH --array=0-7


CM_LIST="/storage/homefs/tj23t050/snRNA_variant_predictions/scripts/taxonomy_paper/cm_model_list.txt"
results_dir="/storage/homefs/tj23t050/snRNA_variant_predictions/results/cmscan_humangenome"
genome_fasta="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/Nam/data/Ensembl/Homo_sapiens.GRCh38.dna.toplevel.fa"

mkdir -p "$results_dir"

# Get the correct CM file for this array task
cm_file=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CM_LIST")
snrna=$(basename "$cm_file" .cm)

echo "[$(date)] Running cmscan for $snrna (task ID: $SLURM_ARRAY_TASK_ID)..."

# Run cmscan
cmscan \
  --cpu 4 \
  --cut_ga \
  --tblout "$results_dir/${snrna}.tblout" \
  "$cm_file" \
  "$genome_fasta"

echo "[$(date)] Finished cmscan for $snrna."
