# Computational Analysis of Minor snRNA and its Role in Prostate Cancer

End-to-end analysis of human snRNA promoters and nearby regions:
- **Covariance model scanning** across the genome (Infernal `cmscan`)
- **Population genetics** (gnomAD subpopulation AF, enrichment, Z-scores)
- **GWAS overlaps** (promoters / gene bodies)
- **Expression summaries** from STAR `ReadsPerGene.out.tab` for prostate cell lines
- **Promoter classification** with DNABERT-2 (binary + multiclass)
- **Model interpretation** (token/hidden activations)
- **Motif discovery/quantification** (FIMO; canonical motif heatmaps)

---

## Contents

### Expression summaries
- Aggregates STAR `ReadsPerGene.out.tab` across replicates
- Maps Ensembl IDs → snRNA names, computes per–cell line means
- Grouped barplots for **U4atac/U6atac** in prostate lines (LNCaP, 22RV1, PM154)

### Covariance models (Infernal)
- SLURM array to run `cmscan` per model across GRCh38
- Tabular outputs (`.tblout`), suggested post-processing

### Population & GWAS
- gnomAD parsing (AF/AN per subpopulation), enrichment by std/Z-score
- GWAS overlaps with promoter/gene BEDs
- Stacked/pop-annotated plots of AF & sample sizes

### Motif landscape (MEME Suite FIMO)
- SLURM job to scan upstream/downstream windows with JASPAR PFMs
- Collapsing hits in −200..+50 promoter window, canonical motif sets per snRNA class
- Prevalence barplots and **presence/absence heatmaps** per class

### DNABERT-2 models & inference
- Training scripts for several datasets (e.g. `prom_300_tata`, `prom_core_all`, etc.)
- Validation utilities (accuracy/F1, confusion matrix, violin plots)
- Prediction pipelines for arbitrary FASTA/CSV sequence lists
- **Hidden activation extraction** + per-class top tokens

---

## Quickstart

### Environment (conda)
To recreate the conda environment use command:
conda env create -f environment_full.yml
