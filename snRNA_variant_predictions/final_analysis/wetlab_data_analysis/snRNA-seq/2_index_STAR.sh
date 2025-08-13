#!/bin/bash

#SBATCH --job-name=star_genome_index
#SBATCH --output=./log/star_genome_index_%j.out
#SBATCH --error=./log/star_genome_index_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=epyc2

mkdir -p ./log

module load STAR/2.7.9a-GCC-10.3.0

# Inputs
gtf="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/Nam/data/Ensembl/gencode.v47.annotation_rmchr.gtf"
genome_fasta="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/Nam/data/Ensembl/Homo_sapiens.GRCh38.dna.toplevel.fa"

# Output directory for STAR index
star_index_dir="/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/Nam/data/star_genome_index"
mkdir -p "$star_index_dir"

echo "Copying FASTA and GTF to TMPDIR..."
cp "$gtf" "$TMPDIR/annotation.gtf"
cp "$genome_fasta" "$TMPDIR/genome.fa"

# Estimate max read length
read_length=112

cd "$TMPDIR"

# Build STAR index
STAR --runThreadN 16 \
     --runMode genomeGenerate \
     --genomeDir "$TMPDIR/star_genome_index" \
     --genomeFastaFiles "$TMPDIR/genome.fa" \
     --sjdbGTFfile "$TMPDIR/annotation.gtf" \
     --sjdbOverhang $((read_length - 1))

# Copy result back to final location
cp -r "$TMPDIR/star_genome_index/"* "$star_index_dir/"

echo "STAR genome index build complete."
