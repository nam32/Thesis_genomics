
"""

Purpose:
-------
1) Load & harmonise gnomAD snRNA frequency TSVs
2) Compute population-enrichment (STD-based) and allele-specific Z-score hits
3) Build position-level enrichments
4) Cross-join STD and Z-score hits
5) Match enriched variants to GWAS by rsID
6) Overlap GWAS with snRNA promoter/gene BEDs
7) Produce summary plots/tables

Notes
-----
- Assumes gnomAD TSVs with columns (after renaming):
  CHROM, POS, rsID, REF, ALT, AF, AN, AF_afr, AN_afr, ... AF_sas, AN_sas, source, gene_id, gene_name
- Assigns snRNA class from `gene_name` (regex; order matters)
- Z-scores handle std==0 safely (masked)

"""

import os
import pandas as pd
import re
from glob import glob

folder_path = "/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/gnomAD/gnomad_vcf_snRNA_only"

tsv_files = glob(os.path.join(folder_path, "*_frequencies.tsv"))

dataframes = []

# Process each file
for tsv_file in tsv_files:
    label = os.path.basename(tsv_file).replace("_frequencies.tsv", "")
    
    # Read the TSV file
    df = pd.read_csv(tsv_file, sep='\t', header=None)
    
    # Add a column for the label
    df['source'] = label
    
    # Append to list
    dataframes.append(df)

# Concatenate all into a single dataframe
combined_df = pd.concat(dataframes, ignore_index=True)

df_filtered = combined_df.copy()

df_filtered.columns = [
    "CHROM", "POS", "rsID", "REF", "ALT", "AF", "AN",
    "AF_afr", "AN_afr",
    "AF_ami", "AN_ami",
    "AF_amr", "AN_amr",
    "AF_asj", "AN_asj",
    "AF_eas", "AN_eas",
    "AF_fin", "AN_fin",
    "AF_nfe", "AN_nfe",
    "AF_sas", "AN_sas",
    "source", "gene_id", "gene_name"
]

df_filtered.to_csv("tsv_data_cleaned.csv", sep="\t", index=False)

########################################################################################

pop_af_cols = [col for col in df_filtered.columns if col.startswith("AF_")]

df_long = df_filtered.melt(
    id_vars=["rsID", "POS", "snRNA_class", "gene_name", "REF", "ALT"],
    value_vars=pop_af_cols,
    var_name="Population",
    value_name="AF_subpop"
)

# Enrichment metrics
enrichment = df_long.groupby(["rsID", "POS", "gene_name", "REF", "ALT"]).agg(
    max_AF=("AF_subpop", "max"),
    std_AF=("AF_subpop", "std")
).reset_index()

# Identify the subpop responsible for max_AF
df_max = df_long.merge(enrichment, on=["rsID", "POS", "gene_name", "REF", "ALT"])
df_max = df_max[df_max["AF_subpop"] == df_max["max_AF"]]

# Filter for significantly enriched variants
df_enriched = df_max[(df_max["max_AF"] > 0.01) & (df_max["std_AF"] < 0.02)]


df_enriched_sorted = df_enriched.sort_values(by=["gene_name", "snRNA_class", "POS", "Population"])
df_matched = df_long.merge(df_enriched[["rsID", "POS"]], on=["rsID", "POS"], how="inner")
df_matched = df_matched.sort_values(by=["gene_name", "snRNA_class", "POS"])

df_matched.to_csv("std_enriched.csv", sep="\t", index=False)

df_enriched_sorted.to_csv("std_enriched_only.csv", sep="\t", index=False)

########################################################################################

# Allele-specific enrichment analysis using Z-score

# Step 1: Compute mean and std dev across subpopulations for each specific allele
enrichment_allele = df_long.groupby(["rsID", "POS", "gene_name", "REF", "ALT"]).agg(
    mean_AF=("AF_subpop", "mean"),
    std_AF=("AF_subpop", "std")
).reset_index()

# Step 2: Merge the stats back to the original long-format DataFrame
df_with_stats = df_long.merge(enrichment_allele, on=["rsID", "POS", "gene_name", "REF", "ALT"])

# Step 3: Compute Z-score per population
df_with_stats["AF_zscore"] = (
    (df_with_stats["AF_subpop"] - df_with_stats["mean_AF"]) / df_with_stats["std_AF"]
)

# Step 4: Filter for population-specific outliers (Z > 2 and AF > 1%)
df_enriched_allele = df_with_stats[
    (df_with_stats["AF_subpop"] > 0.01) & (df_with_stats["AF_zscore"].abs() > 2)
]

# Optional: sort by gene and position
df_enriched_allele = df_enriched_allele.sort_values(by=["gene_name", "snRNA_class", "POS", "AF_zscore"], ascending=[True, True, True, False])

# Optional: export
df_enriched_allele.to_csv("zscore_enriched_allele_specific.tsv", sep="\t", index=False)


########################################################################################
# position level
# Reshape to long format, include gene_name
df_long_position_only = df_filtered.melt(
    id_vars=["rsID", "POS", "snRNA_class", "gene_name"],
    value_vars=pop_af_cols,
    var_name="Population",
    value_name="AF_subpop"
)


enrichment_position = df_long_position_only.groupby(["rsID", "POS", "gene_name"]).agg(
    max_AF=("AF_subpop", "max"),
    std_AF=("AF_subpop", "std")
).reset_index()


# Identify the subpop responsible for max_AF
df_max_position = df_long_position_only.merge(enrichment_position, on=["rsID", "POS", "gene_name"])
df_max_position = df_max_position[df_max_position["AF_subpop"] == df_max_position["max_AF"]]

# Filter for significantly enriched variants
df_enriched_position = df_max_position[(df_max_position["max_AF"] > 0.01) & (df_max_position["std_AF"] < 0.02)]


df_enriched_position_sorted = df_enriched_position.sort_values(by=["gene_name", "snRNA_class", "POS", "Population"])
#df_matched_position = df_long.merge(df_enriched_position[["rsID", "POS"]], on=["rsID", "POS"], how="inner")
#df_matched_position = df_matched_position.sort_values(by=["gene_name", "snRNA_class", "POS"])

df_enriched_position_sorted.to_csv("std_enriched_position_only.csv", sep="\t", index=False)

########################################################################################

# Compute mean and std per variant position
enrichment_position = df_long_position_only.groupby(["rsID", "POS", "gene_name"]).agg(
    mean_AF=("AF_subpop", "mean"),
    std_AF=("AF_subpop", "std")
).reset_index()

# Merge back
df_with_stats = df_long_position_only.merge(enrichment_position, on=["rsID", "POS", "gene_name"])

# Compute Z-score
df_with_stats["AF_zscore"] = (df_with_stats["AF_subpop"] - df_with_stats["mean_AF"]) / df_with_stats["std_AF"]

# Keep variants with strong deviation
df_enriched_zscore = df_with_stats[
    (df_with_stats["AF_subpop"] > 0.01) &  # optional: skip very rare ones
    (df_with_stats["AF_zscore"].abs() > 2)  # e.g., >2 standard deviations away
]

df_enriched_zscore = df_enriched_zscore.sort_values(by=["gene_name", "snRNA_class", "POS", "Population"])

df_enriched_zscore.to_csv("zscore_enriched_position_only.csv", sep="\t", index=False)

########################################################################################

#df_enriched_sorted
#df_enriched_allele
#df_enriched_position_sorted
#df_enriched_zscore

df_combined_zscore_std = df_enriched_allele.merge(
    df_enriched_sorted,
    on=["rsID", "POS", "gene_name", "REF", "ALT", "Population"],
    how="left"
)

df_combined_zscore_std.to_csv("combined_zscore_enriched_std_only.csv", sep="\t", index=False)

df_combined_zscore_std_position = df_enriched_zscore.merge(
    df_enriched_position_sorted,
    on=["rsID", "POS", "gene_name", "Population"],
    how="left"
)

df_combined_zscore_std_position.to_csv("combined_zscore_enriched_std_position_only.csv", sep="\t", index=False)

########################################################################################


gwas_df = pd.read_csv("/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/GWAS/alternative", sep="\t")

gwas_df["SNP_ID_CURRENT"] = "rs" + gwas_df["SNP_ID_CURRENT"].astype(str)

gwas_df.rename(columns={"SNP_ID_CURRENT": "rsID"}, inplace=True)
gwas_df["rsID"] = gwas_df["rsID"].astype(str).str.replace(r"\.0$", "", regex=True)

gwas_df["chrom"] = gwas_df["CHR_ID"].astype(str)

matched = df_enriched_zscore.merge(gwas_df[["CHR_ID", "CHR_POS", "DISEASE/TRAIT", "P-VALUE", "OR or BETA", "95% CI (TEXT)", "LINK", "STUDY", "PUBMEDID", "FIRST AUTHOR", "JOURNAL", "RISK ALLELE FREQUENCY", "MAPPED_GENE", "UPSTREAM_GENE_ID","rsID"]], on="rsID", how="inner")

matched.to_csv("matched.csv", sep="\t", index=False)

########################################################################################

## GWAS for promoter region
import numpy as np

gwas_df["pos"] = gwas_df["CHR_POS"]

gwas_df["chrom"] = gwas_df["CHR_ID"].astype(str)

bed_columns = ['chrom', 'start', 'end', 'name', 'score', 'strand']

bed_path = '/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results/rfam_ensembl/promoter_regions.bed'
bed_df = pd.read_csv(bed_path, sep='\t', header=None, names=bed_columns)
bed_df["chrom"] = bed_df["chrom"].astype(str)


labels = ["U6atac", "U6", "U4atac", "U4", "U1", "U2", "U5", "U11", "U12", "U7"]
conditions = [
    bed_df["name"].str.contains(r"\bU6ATAC\b|^RNU6ATAC", case=False, na=False),
    bed_df["name"].str.contains(r"\bU6\b|^RNU6(?!ATAC)", case=False, na=False),
    bed_df["name"].str.contains(r"\bU4ATAC\b|^RNU4ATAC", case=False, na=False),
    bed_df["name"].str.contains(r"\bU4\b|^RNU4-", case=False, na=False),
    bed_df["name"].str.contains(r"\bU1\b|^RNU1-|^RNVU1", case=False, na=False),
    bed_df["name"].str.contains(r"\bU2\b|^RNU2", case=False, na=False),
    bed_df["name"].str.contains(r"\bU5\b|^RNU5", case=False, na=False),
    bed_df["name"].str.contains(r"\bU11\b|^RNU11", case=False, na=False),
    bed_df["name"].str.contains(r"\bU12\b|^RNU12", case=False, na=False),
    bed_df["name"].str.contains(r"\bU7\b|^RNU7", case=False, na=False),
]

bed_df["snrna_class"] = np.select(conditions, labels, default="Other")
bed_df = bed_df.sort_values(by="snrna_class").reset_index(drop=True)
gwas_df["pos"] = pd.to_numeric(gwas_df["CHR_POS"], errors="coerce")


overlaps = []
for _, bed_row in bed_df.iterrows():
    chrom = bed_row["chrom"]
    start = bed_row["start"]
    end = bed_row["end"]
    
    overlap_hits = gwas_df[
        (gwas_df["chrom"] == chrom) &
        (gwas_df["pos"] >= start) &
        (gwas_df["pos"] <= end)
    ].copy()
    
    if not overlap_hits.empty:
        overlap_hits["promoter"] = bed_row["name"]
        overlaps.append(overlap_hits)

# Combine all overlaps into a single DataFrame
if overlaps:
    overlap_df = pd.concat(overlaps, ignore_index=True)
else:
    overlap_df = pd.DataFrame()


conditions = [
    overlap_df["promoter"].str.contains(r"\bU6ATAC\b|^RNU6ATAC", case=False, na=False),
    overlap_df["promoter"].str.contains(r"\bU6\b|^RNU6(?!ATAC)", case=False, na=False),
    overlap_df["promoter"].str.contains(r"\bU4ATAC\b|^RNU4ATAC", case=False, na=False),
    overlap_df["promoter"].str.contains(r"\bU4\b|^RNU4-", case=False, na=False),
    overlap_df["promoter"].str.contains(r"\bU1\b|^RNU1-|^RNVU1", case=False, na=False),
    overlap_df["promoter"].str.contains(r"\bU2\b|^RNU2", case=False, na=False),
    overlap_df["promoter"].str.contains(r"\bU5\b|^RNU5", case=False, na=False),
    overlap_df["promoter"].str.contains(r"\bU11\b|^RNU11", case=False, na=False),
    overlap_df["promoter"].str.contains(r"\bU12\b|^RNU12", case=False, na=False),
    overlap_df["promoter"].str.contains(r"\bU7\b|^RNU7", case=False, na=False),
]

overlap_df["snrna_class"] = np.select(conditions, labels, default="Other")
overlap_df = overlap_df.sort_values(by="snrna_class").reset_index(drop=True)


prostate_gwas = gwas_df[gwas_df['DISEASE/TRAIT'].str.contains('prostate cancer', case=False, na=False)]


minor_snRNA=["U11", "U12", "U6atac", "U4atac"]
overlap_df["snrna_name"] = overlap_df["promoter"].str.split('|').str[0]

minor_overlap_df=overlap_df[overlap_df["snrna_class"].isin(minor_snRNA)]

prostate_gwas.to_csv("/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/GWAS/prostate_gwas.csv", sep="\t", index=False)

gwas_df


########################################################################################

## GWAS for promoter region

gene_bed_path = '/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results/rfam_ensembl/snRNA_gene_gencode_combined_sorted.bed'
gene_bed_df = pd.read_csv(gene_bed_path, sep='\t', header=None, names=bed_columns)
gene_bed_df["chrom"] = gene_bed_df["chrom"].astype(str)


gene_overlaps = []
for _, bed_row in gene_bed_df.iterrows():
    chrom = bed_row["chrom"]
    start = bed_row["start"]
    end = bed_row["end"]
    
    overlap_hits = gwas_df[
        (gwas_df["chrom"] == chrom) &
        (gwas_df["pos"] >= start) &
        (gwas_df["pos"] <= end)
    ].copy()
    
    if not overlap_hits.empty:
        overlap_hits["promoter"] = bed_row["name"]
        gene_overlaps.append(overlap_hits)

# Combine all overlaps into a single DataFrame
if gene_overlaps:
    gene_overlap_df = pd.concat(gene_overlaps, ignore_index=True)
else:
    gene_overlap_df = pd.DataFrame()


conditions = [
    gene_overlap_df["promoter"].str.contains(r"\bU6ATAC\b|^RNU6ATAC", case=False, na=False),
    gene_overlap_df["promoter"].str.contains(r"\bU6\b|^RNU6(?!ATAC)", case=False, na=False),
    gene_overlap_df["promoter"].str.contains(r"\bU4ATAC\b|^RNU4ATAC", case=False, na=False),
    gene_overlap_df["promoter"].str.contains(r"\bU4\b|^RNU4-", case=False, na=False),
    gene_overlap_df["promoter"].str.contains(r"\bU1\b|^RNU1-|^RNVU1", case=False, na=False),
    gene_overlap_df["promoter"].str.contains(r"\bU2\b|^RNU2", case=False, na=False),
    gene_overlap_df["promoter"].str.contains(r"\bU5\b|^RNU5", case=False, na=False),
    gene_overlap_df["promoter"].str.contains(r"\bU11\b|^RNU11", case=False, na=False),
    gene_overlap_df["promoter"].str.contains(r"\bU12\b|^RNU12", case=False, na=False),
    gene_overlap_df["promoter"].str.contains(r"\bU7\b|^RNU7", case=False, na=False),
]

gene_overlap_df["snrna_class"] = np.select(conditions, labels, default="Other")
gene_overlap_df = gene_overlap_df.sort_values(by="snrna_class").reset_index(drop=True)

minor_snRNA=["U11", "U12", "U6atac", "U4atac"]
gene_overlap_df["snrna_name"] = gene_overlap_df["promoter"].str.split('|').str[0]

minor_gene_overlap_df=gene_overlap_df[gene_overlap_df["snrna_class"].isin(minor_snRNA)]

########################################################################################

tsv_df = pd.read_csv("/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/gnomAD/gnomad_vcf_snRNA_only/tsv_data_cleaned.csv", sep="\t")

pop_af_cols = [
    "AF_afr", "AF_ami", "AF_amr", "AF_asj",
    "AF_eas", "AF_fin", "AF_nfe", "AF_sas"
]

# Ensure AF columns are numeric and safely replace zeros with NaN
for col in pop_af_cols:
    tsv_df[col] = pd.to_numeric(tsv_df[col], errors='coerce')
    tsv_df[col] = tsv_df[col].mask(tsv_df[col] == 0.0, np.nan)

tsv_df["AF_std"] = tsv_df[pop_af_cols].std(axis=1, skipna=True)

top_variable = tsv_df.nlargest(20, "AF_std")

stacked_data = []
for _, row in top_variable.iterrows():
    pos_label = f"{row['POS']} ({row['snRNA_class']})"
    for pop in pop_af_cols:
        af = row[pop]
        if pd.isna(af):
            continue
        stacked_data.append({
            "Position": pos_label,
            "Population": pop.replace("AF_", "").upper(),
            "Allele": row["REF"],
            "AF": 1 - af
        })
        stacked_data.append({
            "Position": pos_label,
            "Population": pop.replace("AF_", "").upper(),
            "Allele": row["ALT"],
            "AF": af
        })

melted_df = pd.DataFrame(stacked_data)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot
g = sns.catplot(
    data=melted_df,
    x="Position", y="AF", hue="Allele", col="Population",
    kind="bar", height=4, aspect=1.5, col_wrap=4
)
g.set_xticklabels(rotation=90)
g.set_titles(col_template="{col_name}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Allele Frequency Contributions (A/T/C/G) Across Populations\nTop Variable snRNA SNV Positions", fontsize=14)
plt.tight_layout()
output_path = "/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/gnomAD/gnomad_vcf_snRNA_only/top_variable_snRNA_SNVs_by_population_allele_stackbar.png"

plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

########################################################################################

## check biallele etc SNV 1 substitution

tsv_df_filtered = tsv_df[
    (tsv_df['REF'].str.len() == 1) & (tsv_df['ALT'].str.len() == 1)
]
rsid_counts = tsv_df_filtered[['rsID', 'REF', 'snRNA_class']].value_counts().reset_index(name='count')
rsid_counts_filtered = rsid_counts[rsid_counts['rsID'] != '.']
rsid_counts_filtered = rsid_counts_filtered[rsid_counts_filtered['REF'].str.len() == 1]

count_summary = rsid_counts_filtered.groupby(['snRNA_class', 'count']).size().reset_index(name='occurrences')
count_summary.to_csv("allele_analysis.csv", sep=",", index=False)

########################################################################################

## SNV 1 insertion

tsv_df_filtered = tsv_df[
    (tsv_df['REF'].str.len() < tsv_df['ALT'].str.len())
]
tsv_df_filtered[tsv_df_filtered['REF'].str.len() > 1]
rsid_counts = tsv_df_filtered[['rsID', 'REF', 'snRNA_class']].value_counts().reset_index(name='count')
rsid_counts_filtered = rsid_counts[rsid_counts['rsID'] != '.']

count_summary = rsid_counts_filtered.groupby(['snRNA_class', 'count']).size().reset_index(name='occurrences')

########################################################################################

## deletion

tsv_df_filtered = tsv_df[
    (tsv_df['REF'].str.len() > tsv_df['ALT'].str.len())
]

rsid_counts = tsv_df_filtered[['rsID', 'REF', 'snRNA_class']].value_counts().reset_index(name='count')
rsid_counts_filtered = rsid_counts[rsid_counts['rsID'] != '.']

count_summary = rsid_counts_filtered.groupby(['snRNA_class', 'count']).size().reset_index(name='occurrences')

########################################################################################

rsid_counts_filtered = tsv_df[tsv_df['rsID'] != '.']
num_unique_rsid = rsid_counts_filtered["rsID"].unique().shape[0]

mutations_per_gene_class_unique = (
    tsv_df.drop_duplicates(subset=['CHROM', 'POS', 'gene_id', 'snRNA_class'])
    .groupby(['gene_name', 'snRNA_class'])
    .size()
    .reset_index(name='unique_mutation_count')
)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Jitter plot
plt.figure(figsize=(8, 5))
for i, snrna_class in enumerate(mutations_per_gene_class_unique['snRNA_class'].unique()):
    y = mutations_per_gene_class_unique[mutations_per_gene_class_unique['snRNA_class'] == snrna_class]['unique_mutation_count']
    x = np.random.normal(i + 1, 0.05, size=len(y))  # jitter
    plt.scatter(x, y, marker='x', alpha=0.7)

plt.xticks(range(1, len(mutations_per_gene_class_unique['snRNA_class'].unique()) + 1),
           mutations_per_gene_class_unique['snRNA_class'].unique())
plt.xlabel('snRNA_class')
plt.ylabel('Unique Mutation Count')
plt.title('Unique Mutation Counts per snRNA Class (per Gene)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

output_path = "unique_mutation_counts_jitter.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

########################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Jitter plot
plt.figure(figsize=(8, 5))
for i, snrna_class in enumerate(mutations_per_gene_class_unique['snRNA_class'].unique()):
    subset = mutations_per_gene_class_unique[mutations_per_gene_class_unique['snRNA_class'] == snrna_class]
    y = subset['unique_mutation_count']
    x = np.random.normal(i + 1, 0.05, size=len(y))  # jitter
    plt.scatter(x, y, marker='x', alpha=0.7)
    # Add labels for each point
    for xi, yi, label in zip(x, y, subset['gene_name']):
        plt.text(xi, yi, label, fontsize=8, ha='center', va='bottom')

plt.xticks(
    range(1, len(mutations_per_gene_class_unique['snRNA_class'].unique()) + 1),
    mutations_per_gene_class_unique['snRNA_class'].unique()
)
plt.xlabel('snRNA_class')
plt.ylabel('Unique Mutation Count')
plt.title('Unique Mutation Counts per snRNA Class (per Gene)')
plt.grid(True, linestyle='--', alpha=0.5)

# Save before showing
output_path = "unique_mutation_counts_jitter.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()
