import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
This Python script processes and visualises sequence similarity data between canonical and 
non-canonical snRNA variants using results from two alignment methods: Infernal (with score
-per-nt normalisation) and Jalview (pairwise percent identity). It filters for variant 
sequences that are compared directly against a canonical snRNA (e.g. RNU11, RNU6ATAC) 
and removes canonical sequences from the plotted results. The script then combines both 
datasets, groups them by snRNA type, and generates horizontal strip plots showing the 
percentage identity of each variant to its canonical reference. For each snRNA (U11, U12, 
U6ATAC, U4ATAC), the script saves a separate plot with a top-left legend distinguishing 
between the two data sources.
C:\Users\tjanj\OneDrive\Desktop\Unibe\rubin_lab\inferno_pairwise_per_snRNA_plots
"""

# Canonical identifiers
## target_name	accession	query_name	accession2	mdl	mdl_from	mdl_to	seq_from	seq_to	strand	trunc	pass	gc	bias	score	e_value	inc	description	length	score_per_nt	score_per_nt_norm	snRNA_type	label
## RNU1-1|ENSG00000206652.1::1:16514121-16514285(-)	-	U1	RF00003	cm	1	166	1	164	+	no	1	0.54	0.0	175.1	1.1e-49	!	-	164	1.0676829268292682	100.0	U1	RNU1-1|ENSG00000206652.1

canonical_prefixes = ["RNU11", "RNU12", "RNU6ATAC", "RNU5E-1", "RNU4ATAC"]

def is_canonical(seq):
    return any(seq.startswith(prefix + "|") for prefix in canonical_prefixes)

# Load Infernal dat
df1 = pd.read_csv(
    "/storage/homefs/tj23t050/snRNA_variant_predictions/scripts/0_snRNA_variant_alignment/infernal_alignment_visualisation/combined_infernal_output.tsv",
    sep="\t"
)
df1 = df1.rename(columns={
    "label": "variant",
    "score_per_nt_norm": "percent_identity"
})
df1["source"] = "Infernal (score-per-nt normalised)"

# Extract base gene symbol before "|"
df1["variant"] = df1["variant"].str.split("|").str[0]

# Load Clustal data
df2 = pd.read_csv(
    "/storage/homefs/tj23t050/snRNA_variant_predictions/scripts/0_snRNA_variant_alignment/jal_alignment_visualization/cleaned_summary.txt",
    sep="\t",
    header=None,
    names=["snRNA_type", "seq1", "seq2", "percent_identity"]
)
df2["percent_identity"] = df2["percent_identity"].astype(float)

# Keep only canonical-vs-noncanonical comparisons
df2 = df2[
    (df2["seq1"].apply(is_canonical) & ~df2["seq2"].apply(is_canonical)) |
    (~df2["seq1"].apply(is_canonical) & df2["seq2"].apply(is_canonical))
].copy()

# Extract the non-canonical sequence and remove trailing /1-XXX, then get base gene name
df2["variant"] = df2.apply(
    lambda row: row["seq2"] if is_canonical(row["seq1"]) else row["seq1"],
    axis=1
).str.replace(r"/\d+-\d+$", "", regex=True).str.split("|").str[0]

df2["source"] = "Pairwise Percent Identity"

# Keep only relevant columns
df1 = df1[["snRNA_type", "variant", "percent_identity", "source"]]
df2 = df2[["snRNA_type", "variant", "percent_identity", "source"]]

# Merge and filter
merged_df = pd.concat([df1, df2], ignore_index=True)

outdir = "/storage/homefs/tj23t050/snRNA_variant_predictions/scripts/0_snRNA_variant_alignment/infernal_alignment_visualisation/per_snRNA_plots"
os.makedirs(outdir, exist_ok=True)

target_snRNAs = {"U11", "U12", "U6ATAC", "U4ATAC"}
merged_df = merged_df[merged_df["snRNA_type"].str.upper().isin(target_snRNAs)]
merged_df = merged_df[~merged_df["variant"].isin(canonical_prefixes)]

# Plot
for snrna in sorted(merged_df["snRNA_type"].unique()):
    sub_df = merged_df[merged_df["snRNA_type"] == snrna].copy()

    # Get variant ranking only from pairwise identity values
    pairwise_df = sub_df[sub_df["source"] == "Pairwise Percent Identity"]
    pairwise_order = pairwise_df.sort_values(by="percent_identity", ascending=False)["variant"].unique().tolist()

    # Explicitly assign this order to all variants (including Infernal source)
    sub_df = sub_df[sub_df["variant"].isin(pairwise_order)].copy()
    sub_df["variant"] = pd.Categorical(sub_df["variant"], categories=pairwise_order, ordered=True)

    fig_height = max(2, 0.3 * len(pairwise_order))
    plt.figure(figsize=(10, fig_height))

    sns.stripplot(
        data=sub_df,
        x="percent_identity",
        y="variant",
        hue="source",
        jitter=False,
        size=6,
        orient="h"
    )

    plt.title(f"{snrna} Variant Identity vs Canonical (Infernal & Pairwise Percent Identity)")
    plt.xlabel("Percentage Identity (%)")
    plt.ylabel("Variant")
    plt.xlim(20, 101)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.grid(axis="y", linestyle=":", alpha=0.5)

    plt.legend(
        loc="upper left",
        frameon=True,
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0.5
    )

    plt.tight_layout()
    outfile = os.path.join(outdir, f"{snrna}_ranked_by_pairwise.png")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[✔] Saved: {outfile}")
