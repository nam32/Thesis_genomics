#!/bin/bash

#SBATCH --job-name=STAR_align
#SBATCH --output=./log/STAR_align_%A_%a.out
#SBATCH --error=./log/STAR_align_%A_%a.err
#SBATCH --array=0-11
#SBATCH --time=3:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --partition=epyc2

# Create log directory
mkdir -p ./log

#STAR --version: 2.7.9a
module load STAR/2.7.9a-GCC-10.3.0
module load SAMtools/1.13-GCC-10.3.0

reads_dir="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/Kaitlin_data/snRNA_seq/merged_fastq"
star_index="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/Nam/data/star_genome_index"
output_dir="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/mabin/star_bams_with_counts_50multimap"
mkdir -p "$output_dir"

# Get R1/R2 FASTQ files
r1_files=(${reads_dir}/*_R1.fastq.gz)
r1="${r1_files[$SLURM_ARRAY_TASK_ID]}"
r2="${r1/_R1.fastq.gz/_R2.fastq.gz}"
sample=$(basename "$r1" | sed 's/_R1\.fastq\.gz//')

# Create per-sample scratch directory
scratch_dir="/scratch/local/$SLURM_JOB_ID/$sample"
mkdir -p "$scratch_dir"

echo "[$(date)] Starting alignment for sample: $sample"

# Run STAR
STAR --runThreadN 16 \
     --genomeDir "$star_index" \
     --readFilesIn "$r1" "$r2" \
     --readFilesCommand zcat \
     --outSAMtype BAM SortedByCoordinate \
     --outFilterMultimapNmax 50 \
     --outFileNamePrefix "$scratch_dir/${sample}_" \
     --outTmpDir "$scratch_dir/tmp" \
     --quantMode GeneCounts \
     --outSAMmultNmax 1

# Move STAR outputs to permanent location
mkdir -p "$output_dir/$sample"
mv "$scratch_dir"/* "$output_dir/$sample/"

# Index the BAM file
bam_file="$output_dir/$sample/${sample}_Aligned.sortedByCoord.out.bam"
if [[ -f "$bam_file" ]]; then
    samtools index "$bam_file"
else
    echo "[$(date)] ERROR: BAM file not found for $sample" >&2
    exit 1
fi
