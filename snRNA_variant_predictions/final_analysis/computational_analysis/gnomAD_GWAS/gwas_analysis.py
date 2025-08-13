"""

Purpose:
-------
Load a prostate cancer GWAS table, standardize key fields, derive functional
annotations, and produce gene-level summaries with significance and effect-size
aggregates—plus an RNU-focused subset.

What it does
------------
1) Columns & rename:
   - Keeps: rsID, reported/mapped genes, trait, context, risk allele, P-value,
     effect size (OR/BETA), CI, cohort size, chrom/pos, PubMed ID.
   - Renames to concise, analysis-friendly headers.

2) Functional category:
   - Buckets the “Context” string into coarse classes:
     Coding / UTR / Non-coding / Regulatory / Splicing / Other / Unknown.
     (Case-insensitive; robust to missing values.)

3) Significance & ranking:
   - Numeric coercion for P_value; filters to P>0; computes -log10(P).
   - Sorts records by significance.

4) Gene-level aggregation (by Mapped Gene):
   - SNP_count         : unique rsIDs per gene
   - Min_P_value       : smallest P across SNPs
   - Max_Effect_Size   : max effect size (OR/BETA) observed
   - Mean_Effect_Size  : mean effect size per gene
   - PubMedIDs         : unique PubMed IDs concatenated
   - Adds -log10(Min_P) for ranking.

5) Outputs:
   - `prostate_gene_summary.csv` : all genes, ranked by significance then SNP_count
   - `rnu_gene_summary.csv`      : subset where Mapped Gene contains “RNU”

Assumptions & notes
-------------------
- Input file `/storage/.../GWAS/prostate_gwas.csv` is tab-delimited and already
  harmonized to include: rsID, chrom, pos, Context, etc.
- Effect Size column may mix OR and BETA; aggregation is descriptive only.
- P=0 entries (if any) are removed to avoid infinite -log10(P).
- “RNU” filter is a simple substring check on Mapped Gene.
"""

import os
import pandas as pd
import re
from glob import glob
import numpy as np

prostate_gwas_df = pd.read_csv("/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/GWAS/prostate_gwas.csv", sep="\t")

summary_df = prostate_gwas_df[[
    "rsID", 
    "REPORTED GENE(S)", 
    "MAPPED_GENE", 
    "DISEASE/TRAIT", 
    "CONTEXT", 
    "STRONGEST SNP-RISK ALLELE", 
    "P-VALUE", 
    "OR or BETA", 
    "95% CI (TEXT)",
    "INITIAL SAMPLE SIZE",
    "chrom",
    "pos",
    "PUBMEDID"
]].copy()

summary_df = summary_df.rename(columns={
    "REPORTED GENE(S)": "Gene",
    "MAPPED_GENE": "Mapped Gene",
    "DISEASE/TRAIT": "Trait",
    "CONTEXT": "Context",
    "STRONGEST SNP-RISK ALLELE": "Risk Allele",
    "P-VALUE": "P_value",
    "OR or BETA": "Effect Size",
    "95% CI (TEXT)": "95% CI",
    "INITIAL SAMPLE SIZE": "Cohort Size"
})

def classify_context(context):
    if pd.isna(context):
        return "Unknown"
    context = context.lower()
    if "missense" in context or "stop_gained" in context or "synonymous" in context or "frameshift" in context:
        return "Coding"
    elif "utr" in context:
        return "UTR"
    elif "intron" in context or "non_coding_transcript" in context:
        return "Non-coding"
    elif "intergenic" in context or "tf_binding" in context or "regulatory" in context:
        return "Regulatory"
    elif "splice" in context:
        return "Splicing"
    else:
        return "Other"

summary_df["Functional_Category"] = summary_df["Context"].apply(classify_context)

summary_df["P_value"] = pd.to_numeric(summary_df["P_value"], errors="coerce")
summary_df = summary_df[summary_df["P_value"] > 0].copy()

summary_df["-log10(P)"] = -np.log10(summary_df["P_value"])
summary_df = summary_df.sort_values("-log10(P)", ascending=False)

###############################################################################################################
## Aggregate by mapped genes

# Ensure numeric columns are correctly parsed
summary_df["P_value"] = pd.to_numeric(summary_df["P_value"], errors="coerce")
summary_df["Effect Size"] = pd.to_numeric(summary_df["Effect Size"], errors="coerce")

# Group by "Mapped Gene" and aggregate
gene_summary = summary_df.groupby("Mapped Gene").agg(
    SNP_count = ("rsID", "nunique"),
    Min_P_value = ("P_value", "min"),
    Max_Effect_Size = ("Effect Size", "max"),
    Mean_Effect_Size = ("Effect Size", "mean"),
    PubMedIDs = ("PUBMEDID", lambda x: ', '.join(sorted(set(str(i) for i in x if pd.notna(i)))))
).reset_index()

# Add –log10(P) for better sorting
gene_summary["-log10(Min_P)"] = -np.log10(gene_summary["Min_P_value"])

# Sort by significance or SNP count
gene_summary = gene_summary.sort_values(["-log10(Min_P)", "SNP_count"], ascending=[False, False])

# Filter to only RNU genes
rnu_genes = gene_summary[gene_summary["Mapped Gene"].str.contains("RNU", na=False)]

output_path = "/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/GWAS/rnu_gene_summary.csv"
rnu_genes.to_csv(output_path, index=False)

output_path = "/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/GWAS/prostate_gene_summary.csv"
gene_summary.to_csv(output_path, index=False)

###############################################################################################################


