import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# === Config ===
output_dir = "/storage/homefs/tj23t050/snRNA_variant_predictions/scripts/1_chip-seq_analysis/plot"
os.makedirs(output_dir, exist_ok=True)

# === Load data ===
## conbined df structure:
## biosample_classification,file_id,Strand,Name,cell_line,Start,chip_target,tab_file,signal,Chromosome,End,condition,linked_antibody,chip_type,perturbed,status,cancer_type,snRNA_group
## cell line,ENCFF699QAQ,+,RNU6-778P|ENSG00000200139.1,HepG2,199887658,H3K4me3,ENCFF699QAQ_snRNA_signal_matrix.tab,0.1137427845305669,chr1,199888261,Homo sapiens HepG2,ENCAB000ARB,histone,False,Cancer,Liver,U6

combined_df = pd.read_csv(
    "/storage/homefs/tj23t050/snRNA_variant_predictions/scripts/1_chip-seq_analysis/summary_combined_df_chipseq_2.csv",
    dtype={"accession": str, "file_id": str}
)
combined_df = combined_df[combined_df["perturbed"] == False].copy()
# === snRNA groups and marks ===
snrna_groups = ["U1", "U2", "U4", "U6", "U11", "U12", "U4ATAC", "U6ATAC"]
chip_targets = combined_df["chip_target"].dropna().unique()

# === Cell line ordering ===
ordered_cell_lines = [
    "VCaP", "22Rv1", "LNCaP clone FGC", "PC-3",
    "RWPE1", "RWPE2", "epithelial cell of prostate", "prostate gland",
    "A549", "HeLa-S3", "HepG2", "K562", "MCF-7", "H1"
]

# === Generate plots ===
for snrna_group in snrna_groups:
    for chip_target in chip_targets:
        df = combined_df[(combined_df["chip_target"] == chip_target) & (combined_df["snRNA_group"] == snrna_group)].copy()
        if df.empty:
            continue

        df["variant"] = df["Name"].str.extract(r'^(.*?)(?:\||$)')
        grouped = df.groupby(["variant", "cell_line"])["signal"].mean().reset_index()
        variant_order = grouped.groupby("variant")["signal"].mean().sort_values(ascending=False).index.tolist()
        heatmap_data = grouped.pivot(index="variant", columns="cell_line", values="signal").fillna(0)

        # Reorder rows and columns
        heatmap_data = heatmap_data.loc[variant_order]
        heatmap_data = heatmap_data[[c for c in ordered_cell_lines if c in heatmap_data.columns]]

        # Plot
        plt.figure(figsize=(12, max(6, 0.4 * len(variant_order))))
        sns.heatmap(heatmap_data, cmap="viridis", annot=True, fmt=".2f", linewidths=0.5)
        plt.title(f"{chip_target} Signal for {snrna_group} Variants Across Cell Lines")
        plt.ylabel(f"{snrna_group} Variant")
        plt.xlabel("Cell Line")
        plt.tight_layout()

        # Save
        plot_filename = f"{chip_target}_{snrna_group}_variant_heatmap.png".replace("/", "_")
        plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
        plt.close()
