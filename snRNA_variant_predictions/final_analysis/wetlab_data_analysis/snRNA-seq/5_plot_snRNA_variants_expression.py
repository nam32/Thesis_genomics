"""

Purpose:
-------
1) Collect STAR ReadsPerGene tables from many sample subfolders and merge them.
2) Map Ensembl feature IDs → human-readable snRNA names.
3) Build a (snRNA × sample) count matrix, compute per–cell line means, and
   visualize expression for minor spliceosome snRNAs (U11/U12/U4atac/U6atac).

Inputs
------
- Mapping file: snRNA_mapping.txt  (columns: snRNA_name | Geneid)
- STAR outputs: <base_dir>/<sample>/<sample>_ReadsPerGene.out.tab
  (four columns: gene_id, unstranded, stranded_forward, stranded_reverse)

What the script does
--------------------
• Scans `base_dir` for STAR outputs, reads them (skipping header comments),
  and stacks into one DataFrame with a `sample_dir` column.
• Merges with `snRNA_mapping` to retain only snRNA features;
  builds a `gene_key = gene_id|snRNA_name` for uniqueness.
• Pivots to wide format: rows = gene_key, columns = samples,
  values = **stranded_forward** counts (see note on strandedness below).
• Sorts columns, ranks snRNAs by total counts, then splits the index back into
  ['gene_id', 'snRNA_name'] for readability.
• Derives per–cell line means by collapsing technical replicates inferred from
  column prefixes (prefix = text before trailing _<number>).
• Filters to **minor** snRNA classes via regex (U6ATAC, U4ATAC, U11, U12),
  ranks by across-cell-line mean, and prepares tidy data.
• Plots grouped bar charts for U6atac and U4atac across the three prostate lines
  (LNCaP, 22RV1, PM154), saving PNGs.

Outputs
-------
- Printed file inventory (first few STAR tables found)
- Head of the sorted snRNA count matrix (rows: gene_id|snRNA_name)
- Figures:
  • u6atac_snRNA_expression_grouped.png
  • u4atac_snRNA_expression_grouped.png
"""

import pandas as pd
import re
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

snRNA_mapping = pd.read_csv(
    "/storage/homefs/tj23t050/snRNA_variant_predictions/results/1_featurecount_relaxed/snRNA_mapping.txt",
    sep="|", header=None, names=["snRNA_name", "Geneid"]
)

snRNA_mapping.rename(
    columns={
        snRNA_mapping.columns[0]: "snRNA_name",
        snRNA_mapping.columns[1]: "Geneid"
    },
    inplace=True
)

base_dir = "/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/mabin/star_bams_with_counts_50multimap"
dfs = []

for sample_dir in os.listdir(base_dir):
    full_dir = os.path.join(base_dir, sample_dir)
    if os.path.isdir(full_dir):
        file_path = os.path.join(full_dir, f"{sample_dir}_ReadsPerGene.out.tab")
        if os.path.isfile(file_path):
            # Read file (STAR format: 4 columns: gene_id, unstranded, strand1, strand2)
            df = pd.read_csv(file_path, sep='\t', header=None, comment='#')
            df.columns = ['gene_id', 'unstranded', 'stranded_forward', 'stranded_reverse']
            df["sample_dir"] = sample_dir
            dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
snrna_df = combined_df.merge(snRNA_mapping, left_on="gene_id", right_on="Geneid", how="inner")
snrna_df = snrna_df.drop(columns=["Geneid"])
snrna_df["gene_key"] = snrna_df["gene_id"] + "|" + snrna_df["snRNA_name"]

snrna_matrix = snrna_df.pivot_table(
    index='gene_key',
    columns='sample_dir',
    values='stranded_forward',
    fill_value=0
)

file_info_list = []

for sample_dir in os.listdir(base_dir):
    full_sample_path = os.path.join(base_dir, sample_dir)
    if os.path.isdir(full_sample_path):
        file_path = os.path.join(full_sample_path, f"{sample_dir}_ReadsPerGene.out.tab")
        if os.path.isfile(file_path):
            file_info_list.append({
                "sample_dir": sample_dir,
                "filename": f"{sample_dir}_ReadsPerGene.out.tab",
                "full_path": file_path
            })

df = pd.DataFrame(file_info_list)

snrna_matrix = snrna_matrix.sort_index(axis=1)
snrna_matrix['total_count'] = snrna_matrix.sum(axis=1)
snrna_matrix_sorted = snrna_matrix.sort_values(by='total_count', ascending=False)
snrna_matrix_sorted = snrna_matrix_sorted.drop(columns='total_count')

snrna_matrix_sorted.index = snrna_matrix_sorted.index.str.split('|', expand=True)
snrna_matrix_sorted.index.names = ['gene_id', 'snRNA_name']


sample_cols = snrna_matrix_sorted.columns
cell_lines = sorted(set(re.sub(r'_\d+$', '', col) for col in sample_cols))

cell_line_means = {}
for cl in cell_lines:
    matching_cols = [col for col in sample_cols if col.startswith(cl)]
    if matching_cols:
        cell_line_means[cl] = snrna_matrix_sorted[matching_cols].mean(axis=1)

snrna_means_by_cell_line = pd.DataFrame(cell_line_means)
snrna_means_by_cell_line.index = snrna_matrix_sorted.index
snrna_means_by_cell_line.index.names = ['gene_id', 'snRNA_name']

snrna_means_by_cell_line_reset = snrna_means_by_cell_line.reset_index()

df_minor = snrna_means_by_cell_line.copy().reset_index()

minor_conditions = [
    df_minor["snRNA_name"].str.contains(r"\bU6ATAC\b|^RNU6ATAC", case=False, na=False),
    df_minor["snRNA_name"].str.contains(r"\bU4ATAC\b|^RNU4ATAC", case=False, na=False),
    df_minor["snRNA_name"].str.contains(r"\bU11\b|^RNU11", case=False, na=False),
    df_minor["snRNA_name"].str.contains(r"\bU12\b|^RNU12", case=False, na=False),
]
minor_mask = minor_conditions[0]
for cond in minor_conditions[1:]:
    minor_mask |= cond

df_minor_filtered = df_minor[minor_mask].copy()

df_minor_filtered["total_mean"] = df_minor_filtered[["22RV1", "LNCaP", "PM154"]].mean(axis=1)
df_minor_filtered = df_minor_filtered.sort_values("total_mean", ascending=False).drop(columns="total_mean")

df_melted = df_minor_filtered.melt(
    id_vars=["snRNA_name"],
    value_vars=["22RV1", "LNCaP", "PM154"],
    var_name="Cell Line",
    value_name="Mean Count"
)

df_melted["snRNA_name"] = pd.Categorical(df_melted["snRNA_name"], categories=df_minor_filtered["snRNA_name"], ordered=True)

hue_order = ["LNCaP", "22RV1", "PM154"]
palette = {
    "LNCaP": "#1f77b4",
    "22RV1": "#ff7f0e",
    "PM154": "#2ca02c"
}

df_u6atac = df_minor_filtered[df_minor_filtered["snRNA_name"].str.contains(r"\bU6ATAC\b|^RNU6ATAC", case=False, na=False)].copy()
df_u4atac = df_minor_filtered[df_minor_filtered["snRNA_name"].str.contains(r"\bU4ATAC\b|^RNU4ATAC", case=False, na=False)].copy()

df_u6atac = df_minor_filtered[
    df_minor_filtered["snRNA_name"].str.contains(r"\bU6ATAC\b|^RNU6ATAC", case=False, na=False)
].copy()

df_u6atac = df_u6atac[(df_u6atac[["22RV1", "LNCaP", "PM154"]].sum(axis=1)) > 0]

df_u4atac = df_minor_filtered[
    df_minor_filtered["snRNA_name"].str.contains(r"\bU4ATAC\b|^RNU4ATAC", case=False, na=False)
].copy()

df_u4atac = df_u4atac[(df_u4atac[["22RV1", "LNCaP", "PM154"]].sum(axis=1)) > 0]

def plot_minor_snRNA(df_subset, title, filename):

    df_melt = df_subset.melt(
        id_vars=["snRNA_name"],
        value_vars=["22RV1", "LNCaP", "PM154"],
        var_name="Cell Line",
        value_name="Mean Count"
    )

    df_melt["snRNA_name"] = pd.Categorical(df_melt["snRNA_name"], categories=df_subset["snRNA_name"], ordered=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df_melt,
        x="snRNA_name",
        y="Mean Count",
        hue="Cell Line",
        hue_order=hue_order,
        palette=palette,
        dodge=True
    )
    plt.title(title)
    plt.xlabel("snRNA Name")
    plt.ylabel("Mean Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

plot_minor_snRNA(
    df_u6atac,
    "U6atac Expression in Prostate Cancer Cell Lines",
    "u6atac_snRNA_expression_grouped.png"
)

plot_minor_snRNA(
    df_u4atac,
    "U4atac Expression in Prostate Cancer Cell Lines",
    "u4atac_snRNA_expression_grouped.png"
)
