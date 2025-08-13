for fasta in *.fa; do
    base=$(basename "$fasta" .fa)
    bedfile="${base}.bed"

    grep "^>" "$fasta" | sed 's/^>//' | \
    awk '
    {
        if (match($0, /(.*)\|\S+::([^:]+):([0-9]+)-([0-9]+)\(([+-])\)/, a)) {
            name = a[1]
            chrom = a[2]
            start = a[3]
            end = a[4]
            strand = a[5]
            print chrom "\t" start "\t" end "\t" name "\t0.0\t" strand
        } else {
            print "⚠️ Failed to parse: " $0 > "/dev/stderr"
        }
    }' > "$bedfile"

    echo "✅ Converted $fasta → $bedfile"
done

#########################################
### find new snRNA sequence in cmscan ###
#########################################

module load BEDTools/2.30.0-GCC-10.3.0

cmscan_dir="/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_cmscan_humangenome"
ensembl_dir="/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results_separate_by_snRNA_types"

output_dir="${cmscan_dir}/novel_hits"
mkdir -p "$output_dir"

for file in "$cmscan_dir"/*.bed; do
  base=$(basename "$file")
  ensembl_bed="${ensembl_dir}/${base}"

  if [[ -f "$ensembl_bed" ]]; then
    bedtools intersect -v -a "$file" -b "$ensembl_bed" > "${output_dir}/${base%.bed}_novel.bed"
  else
    echo "Warning: No matching Ensembl BED for $base"
  fi
done

###########################################
### find new snRNA sequence in cmsearch ###
###########################################

#!/usr/bin/env bash

module load BEDTools/2.30.0-GCC-10.3.0

cmsearch_dir="/storage/homefs/tj23t050/snRNA_variant_predictions/scripts/0_snRNA_list/0425_homo_sapiens_blastndb_major_minor_snRNA"
ensembl_dir="/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results_separate_by_snRNA_types"
output_dir="${cmsearch_dir}/novel_hits"
mkdir -p "$output_dir"

for file in "$cmsearch_dir"/*.bed; do
  base=$(basename "$file")
  ensembl_bed="${ensembl_dir}/${base}"

  if [[ -f "$ensembl_bed" ]]; then
    bedtools intersect -v -s -a "$file" -b "$ensembl_bed" > "${output_dir}/${base%.bed}_novel.bed"
  else
    echo "⚠️  Warning: No matching Ensembl BED for $base"
  fi
done

#################################################

#!/usr/bin/env bash

module load BEDTools/2.30.0-GCC-10.3.0

cmsearch_dir="/storage/homefs/tj23t050/snRNA_variant_predictions/scripts/0_snRNA_list/0425_homo_sapiens_blastndb_major_minor_snRNA"
ensembl_dir="/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results_separate_by_snRNA_types"
output_dir="${cmsearch_dir}/novel_hits"
mkdir -p "$output_dir"

shopt -s nocasematch  # Enable case-insensitive matching

for cm_file in "$cmsearch_dir"/*_cmsearch.bed; do
  base=$(basename "$cm_file" _cmsearch.bed)

  # Look for a case-insensitive match in the Ensembl BED directory
  match=""
  for ensembl_bed in "$ensembl_dir"/*.bed; do
    ens_base=$(basename "$ensembl_bed" .bed)
    if [[ "$ens_base" == "$base" ]]; then
      match="$ensembl_bed"
      break
    fi
  done

  if [[ -n "$match" ]]; then
    echo "Comparing $cm_file ↔ $match"
    bedtools intersect -v -s -a "$cm_file" -b "$match" > "${output_dir}/${base}_novel.bed"
  else
    echo "⚠️  Warning: No matching Ensembl BED for $base (case-insensitive match failed)"
  fi
done

shopt -u nocasematch  # Restore default matching behavior
