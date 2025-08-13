"""

Purpose:
-------
Characterize transcription factor motif landscapes around snRNA loci by integrating
FIMO hits with genomic intervals, grouping snRNAs (U1/U2/U4/U5/U6/U11/U12/U4ATAC/U6ATAC),
deriving canonical motif sets per group, normalizing by paralog counts, and plotting
group-wise motif prevalence and presence/absence heatmaps.

Inputs
------
BEDs (GRCh38, 6 columns):
  - snRNA_gene_gencode_combined_sorted.bed                 # gene body
  - snRNA_gene_gencode_combined_sorted_upstream_1000.bed   # -1000..0
  - snRNA_gene_gencode_combined_sorted_downstream_1000.bed  # +1..+1000

FIMO TSVs (MEME Suite; with 'sequence_name', 'start', 'stop', 'motif_id', 'motif_alt_id', 'score', ...):
  - fimo_ensembl_final_combine_snRNA_sorted_fimo/fimo_corrected.tsv
  - fimo_snRNA_gene_gencode_combined_sorted_upstream_1000_fimo_ver/fimo_fixed.tsv
  - fimo_1000nt_downstream_ensembl_final_combine_snRNA_sorted_fimo/fimo_corrected.tsv

Optional metadata:
  - jaspar_class_family_df.csv  # mapping of motif_id → (class, family)

Key conventions & assumptions
-----------------------------
- FASTA headers used for FIMO scans are formatted as "chr_begin_end"; the script
  splits 'sequence_name' on '_' to recover coordinates.
- Coordinates are GRCh38; BED begin/end are cast to int.
- Relative positions:
  * Upstream: rel = start - 1000 (so -1000..0 upstream → -1000..0 rel)
  * Gene body: strand-aware: (+) rel = [start, stop]; (–) rel = [L - stop, L - start]
  * Downstream: rel_start = start + 1 (1-based downstream)
- Motif window for promoter analysis: rel_start ∈ [-200, +50].
- Binning & collapse: 4-nt bins; within each (sequence_name, bin) keep the
  highest-scoring hit to avoid dense near-duplicates.
- snRNA grouping is inferred from the BED name with regex (U6ATAC before U6).

Pipeline outline
----------------
1) Load BEDs and FIMO TSVs for gene body, upstream (−1000..0), and downstream (+1..+1000);
   parse 'sequence_name' into chr/begin/end; cast to int; compute strand-aware
   relative coordinates as above.

2) Merge FIMO hits with the corresponding BED interval (inner join on chr/begin/end)
   to carry snRNA names and strand.

3) Assign each snRNA to a 'grouping' (U1/U2/U4/U5/U6/U11/U12/U4ATAC/U6ATAC/U7/Unknown)
   via regex on the BED 'name'.

4) Promoter window selection: keep upstream hits with rel_start in [-200, +50];
   bin by 4 nt and collapse to the top-scoring hit per bin.

5) (Optional) Annotate motifs with JASPAR class/family by joining on
   jaspar_class_family_df.csv; a web-scraping fallback is included but an official
   JASPAR dump/API is preferred.

6) Define "canonical" snRNAs per group (e.g., U1→RNU1-1, U2→RNU2-1, U6ATAC→RNU6ATAC);
   collect the set of motif_alt_id observed upstream of these canonical loci.

7) Count, per group, how many distinct snRNA variants contain each canonical motif
   (unique BED 'name' count after collapsing). Produce:
   - Raw counts per (motif_alt_id, grouping).
   - Normalized counts = count / (# unique snRNAs in that group).
   - Percent = normalized × 100.

8) Plot, for each group, a bar chart of % snRNAs with each canonical motif
   (saved under .../plots/promoter/<group>_promoter_motif_barplot.png).

9) Produce presence/absence heatmaps (snRNA × motif) for canonical motifs:
   - Version A: only snRNAs with ≥1 canonical motif.
   - Version B: all snRNAs in the group (zero-filled).
   Saved under ./canonical_motif_heatmaps/.

10) Convenience: extract HMGA1 hits and list snRNA names per group; export a
    flat CSV of upstream motifs used in plots.

Outputs
-------
- upstream_200_50_motifs.csv                                  # filtered & collapsed promoter hits
- <group>_promoter_motif_barplot.png                           # per-group prevalence bars
- canonical_motif_heatmaps/<group>_canonical_motifs_*.png      # heatmaps (canonical-only / all)
- jaspar_class_family_df.csv                                   # (if generated)
- Console prints: HMGA1 carrier lists per group

Gotchas & small fixes
---------------------
- Heatmap filenames: this script generates two heatmaps per group; give them
  distinct names (e.g., '{group}_canonical_only.png' vs '{group}_all.png') to
  avoid overwriting.
- Integer binning: Python floor division with negatives (//) floors toward −∞.
  This is fine for fixed-width bins but be aware when interpreting bin labels.
- Canonical list: double-check entries like 'RNU-1' vs 'RNU1' and 'U6' vs 'RNU6-1'
  to match your BED 'snrna_name' values exactly.
- If FASTA headers contain extra underscores, splitting 'sequence_name' on '_'
  may misparse; consider using a structured header or a regex split with validation.
- Web scraping JASPAR can be slow/brittle; prefer the official JASPAR metadata
  TSV/JSON to populate class/family.

"""

import pandas as pd
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

###################################################################################################################################################################################################################################################

downstream_bed=pd.read_csv("./snRNA_gene_gencode_combined_sorted_downstream_1000.bed", sep="\t", comment='#')
downstream_bed.columns = ["chr", "begin", "end", "name", "score", "strand"]
downstream_df = pd.read_csv("C:/Users/tjanj/OneDrive/Desktop/Unibe/rubin_lab/1_fimo/fimo_1000nt_downstream_ensembl_final_combine_snRNA_sorted_fimo/fimo_corrected.tsv", sep="\t", comment='#')
downstream_df[['chr', 'begin', 'end']] = downstream_df['sequence_name'].str.split('_', n=2, expand=True)
downstream_df["begin"] = downstream_df["begin"].astype(int)
downstream_df["end"] = downstream_df["end"].astype(int)
downstream_bed["begin"] = downstream_bed["begin"].astype(int)
downstream_bed["end"] = downstream_bed["end"].astype(int)

############################################################################################

upstream_bed=pd.read_csv("./snRNA_gene_gencode_combined_sorted_upstream_1000.bed", sep="\t", comment='#')
upstream_bed.columns = ["chr", "begin", "end", "name", "score", "strand"]
upstream_df = pd.read_csv("C:/Users/tjanj/OneDrive/Desktop/Unibe/rubin_lab/1_fimo/fimo_snRNA_gene_gencode_combined_sorted_upstream_1000_fimo_ver/fimo_fixed.tsv", sep="\t", comment='#')
upstream_df[['chr', 'begin', 'end']] = upstream_df['sequence_name'].str.split('_', n=2, expand=True)
upstream_df["begin"] = upstream_df["begin"].astype(int)
upstream_df["end"] = upstream_df["end"].astype(int)
upstream_bed["begin"] = upstream_bed["begin"].astype(int)
upstream_bed["end"] = upstream_bed["end"].astype(int)

############################################################################################

snRNA_bed=pd.read_csv("./snRNA_gene_gencode_combined_sorted.bed", sep="\t", comment='#')
snRNA_bed.columns = ["chr", "begin", "end", "name", "score", "strand"]
snRNA = pd.read_csv("C:/Users/tjanj/OneDrive/Desktop/Unibe/rubin_lab/1_fimo/fimo_ensembl_final_combine_snRNA_sorted_fimo/fimo_corrected.tsv", sep="\t", comment='#')
snRNA[['chr', 'begin', 'end']] = snRNA['sequence_name'].str.split('_', n=2, expand=True)
snRNA["begin"] = snRNA["begin"].astype(int)
snRNA["end"] = snRNA["end"].astype(int)
snRNA_bed["begin"] = snRNA_bed["begin"].astype(int)
snRNA_bed["end"] = snRNA_bed["end"].astype(int)

######################
## get relative pos ##
######################

upstream_df["rel_start"] = upstream_df["start"].apply(lambda x: x - 1000)
upstream_df["rel_stop"] = upstream_df["stop"].apply(lambda x: x - 1000)

def get_rel_within_snRNA(row):
    gene_len = row["end"] - row["begin"]
    if row["strand"] == "+":
        return pd.Series({
            "rel_start": row["start"],
            "rel_stop": row["stop"]
        })
    else:
        return pd.Series({
            "rel_start": gene_len - row["stop"],
            "rel_stop": gene_len - row["start"]
        })

snRNA[["rel_start", "rel_stop"]] = snRNA.apply(get_rel_within_snRNA, axis=1)

downstream_df["rel_start"] = downstream_df["start"] + 1
downstream_df["rel_stop"] = downstream_df["stop"]

############################################################################################

merged_upstream_df = pd.merge(
    upstream_bed,
    upstream_df,
    on=["chr", "begin", "end"],
    how="inner"
)
merged_downstream_df = pd.merge(
    downstream_bed,
    downstream_df,
    on=["chr", "begin", "end"],
    how="inner"
)
merged_snRNA_df = pd.merge(
    snRNA_bed,
    snRNA,
    on=["chr", "begin", "end"],
    how="inner"
)

#######################
## UPSTREAM ANALYSIS ##
#######################

merged_upstream_df["grouping"] = "Unknown"

conditions = [
    merged_upstream_df["name"].str.contains(r"\bU6ATAC\b|^RNU6ATAC", case=False, na=False),  # U6ATAC
    merged_upstream_df["name"].str.contains(r"\bU6\b|^RNU6(?!ATAC)", case=False, na=False),  # U6 but not U6ATAC
    merged_upstream_df["name"].str.contains(r"\bU4ATAC\b|^RNU4ATAC", case=False, na=False),  # U4ATAC
    merged_upstream_df["name"].str.contains(r"\bU4\b|^RNU4-", case=False, na=False),         # U4
    merged_upstream_df["name"].str.contains(r"\bU1\b|^RNU1-|^RNVU1", case=False, na=False),  # U1
    merged_upstream_df["name"].str.contains(r"\bU2\b|^RNU2", case=False, na=False),          # U2
    merged_upstream_df["name"].str.contains(r"\bU5\b|^RNU5", case=False, na=False),          # U5
    merged_upstream_df["name"].str.contains(r"\bU11\b|^RNU11", case=False, na=False),        # U11
    merged_upstream_df["name"].str.contains(r"\bU12\b|^RNU12", case=False, na=False),        # U12
    merged_upstream_df["name"].str.contains(r"\bU7\b|^RNU7", case=False, na=False)           # U7
]


choices = ["U6ATAC", "U6", "U4ATAC", "U4", "U1", "U2", "U5", "U11", "U12", "U7"]

merged_upstream_df["grouping"] = np.select(conditions, choices, default="Unknown")

upstream_bed["grouping"] = "Unknown"
conditions_upstream_bed = [
    upstream_bed["name"].str.contains(r"\bU6ATAC\b|^RNU6ATAC", case=False, na=False),
    upstream_bed["name"].str.contains(r"\bU6\b|^RNU6(?!ATAC)", case=False, na=False),
    upstream_bed["name"].str.contains(r"\bU4ATAC\b|^RNU4ATAC", case=False, na=False),
    upstream_bed["name"].str.contains(r"\bU4\b|^RNU4-", case=False, na=False),
    upstream_bed["name"].str.contains(r"\bU1\b|^RNU1-|^RNVU1", case=False, na=False),
    upstream_bed["name"].str.contains(r"\bU2\b|^RNU2", case=False, na=False),
    upstream_bed["name"].str.contains(r"\bU5\b|^RNU5", case=False, na=False),
    upstream_bed["name"].str.contains(r"\bU11\b|^RNU11", case=False, na=False),
    upstream_bed["name"].str.contains(r"\bU12\b|^RNU12", case=False, na=False),
    upstream_bed["name"].str.contains(r"\bU7\b|^RNU7", case=False, na=False)
]

upstream_bed["grouping"] = np.select(conditions_upstream_bed, choices, default="Unknown")

############################################################################################

## filter for -200 to -50 and collapse

upstream_200_50_filtered_df = merged_upstream_df[
    (merged_upstream_df["rel_start"] >= -200) & (merged_upstream_df["rel_start"] <= 50)
]

# Make sure rel_start is sorted
upstream_200_50_filtered_df = upstream_200_50_filtered_df.sort_values(by=["sequence_name", "motif_alt_id", "rel_start", "score_y"], ascending=[True, True, True, False])

############################################################################################

bin_size = 4
upstream_200_50_filtered_df["rel_bin"] = upstream_200_50_filtered_df["rel_start"] // bin_size

# Sort by best scoring hit per bin
upstream_200_50_filtered_df_collapsed_df = (
    upstream_200_50_filtered_df
    .sort_values(by=["sequence_name", "rel_bin", "score_y"], ascending=[True, True, False])
    .groupby(["sequence_name", "rel_bin"], as_index=False)
    .first()
    .drop(columns="rel_bin")
)

###########
## Class ##
###########

jaspar_class_family_df = pd.read_csv("jaspar_class_family_df.csv", index_col=0)

#jaspar_metadata = pd.read_csv("jaspar_metadata.tsv", sep="\t")

upstream_200_50_filtered_df_collapsed_df = pd.merge(
    upstream_200_50_filtered_df_collapsed_df,
    jaspar_class_family_df,
    left_on=['motif_id'],
    right_on=['matrix_id'],
    how='left'  # use 'left' to preserve all original rows
)

############################################################################################

## canonical snRNA
# Define the pattern — names starting with any of these

upstream_200_50_filtered_df_collapsed_df["snrna_name"] = upstream_200_50_filtered_df_collapsed_df["name"].str.split("|").str[0]

# Step 1: Define canonical snRNAs per group
canonical_snRNAs_per_group = {
    "U4": ["RNU4-1"],
    "U6ATAC": ["RNU6ATAC"],
    "U4ATAC": ["RNU4ATAC"],
    "U1": ["RNU1-1"],
    "U2": ["RNU2-1"],
    "U5": ["RNU5A-1"],
    "U11": ["RNU11"],
    "U12": ["RNU12"],
    "U7": ["RNU7-1"],
    "U6": ["RNU6-1"]
    # Add more as needed
}

# Step 2: Extract motifs found upstream of these canonical snRNAs
canonical_motif_rows = []

for group, snrna_list in canonical_snRNAs_per_group.items():
    # Filter for snRNAs in this group
    matched_rows = upstream_200_50_filtered_df_collapsed_df[
        upstream_200_50_filtered_df_collapsed_df["snrna_name"].isin(snrna_list)
    ]
    # Get unique motif_alt_id values
    motif_names = matched_rows["motif_alt_id"].dropna().unique()
    
    for motif in motif_names:
        canonical_motif_rows.append({"motif_alt_id": motif, "grouping": group})
merged_upstream_df_canonical_motifs = pd.DataFrame(canonical_motif_rows).drop_duplicates()

###############################
## motif count for all snRNA ##
###############################

upstream_motif_snRNA_counts = (
    upstream_200_50_filtered_df_collapsed_df
    .groupby(["motif_alt_id", "grouping"])["name"]
    .nunique()
    .reset_index(name="snRNA_count")
)

group_order = ["U1", "U2", "U4", "U5", "U6", "U7", "U11", "U12", "U4ATAC", "U6ATAC", "Unknown"]

upstream_motif_snRNA_counts["grouping"] = pd.Categorical(
    upstream_motif_snRNA_counts["grouping"],
    categories=group_order,
    ordered=True
)

############################################################################################

upstream_motif_snRNA_counts = upstream_motif_snRNA_counts.sort_values(by="grouping").reset_index(drop=True)

# only counting those that appear in canonical
upstream_motif_snRNA_counts_canonical_motifs_only = pd.merge(
    upstream_motif_snRNA_counts,
    merged_upstream_df_canonical_motifs[["motif_alt_id", "grouping"]],
    on=["motif_alt_id", "grouping"],
    how="inner"
)

upstream_motif_snRNA_counts_canonical_motifs_only = upstream_motif_snRNA_counts_canonical_motifs_only.sort_values(by=["grouping", "snRNA_count"], ascending=[True, False])

###############
## NORMALIZE ##
###############

# calculate total of each snRNA
snRNA_by_types_paralog_counts = (
    upstream_bed[["name", "grouping"]]
    .drop_duplicates()
    .groupby("grouping")
    .size()
    .reset_index(name="unique_snRNA_count")
    .sort_values(by="grouping")
)

############################################################################################

# Merge to normalize
normalized_counts = pd.merge(
    upstream_motif_snRNA_counts_canonical_motifs_only,
    snRNA_by_types_paralog_counts,
    on="grouping",
    how="left"
)

# Add normalized column
normalized_counts["normalized_snRNA_count"] = normalized_counts["snRNA_count"] / normalized_counts["unique_snRNA_count"]

normalized_counts["percent_snRNAs_with_motif"] = (
    normalized_counts["snRNA_count"] / normalized_counts["unique_snRNA_count"] * 100
).round(2)

##########
## PLOT ##
##########
import os

output_dir = r"C:/Users/tjanj/OneDrive/Desktop/Unibe/rubin_lab/plots/promoter"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

groupings = ["U1", "U2", "U4", "U6", "U11", "U12", "U4ATAC", "U6ATAC"]

for group in groupings:
    group_data = normalized_counts[normalized_counts["grouping"] == group].sort_values(
        by="percent_snRNAs_with_motif", ascending=False
    )

    if not group_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=group_data,
            x="motif_alt_id",
            y="percent_snRNAs_with_motif",
            color="#1f77b4"  # single colour (matplotlib default blue)
        )
        plt.xticks(rotation=60, ha='right')
        plt.title(f"{group}", fontsize=14)
        plt.xlabel("Motif ID", fontsize=12)
        plt.ylabel("% with motif", fontsize=12)
        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(output_dir, f"{group}_promoter_motif_barplot.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

############################################################################################

# Filter to HMGA1 motifs
hmga1_hits = upstream_200_50_filtered_df_collapsed_df[
    upstream_200_50_filtered_df_collapsed_df["motif_alt_id"] == "Hmga1"
]

# Get unique snRNA variant names
hmga1_snRNA_variants = hmga1_hits["name"].unique()

hmga1_snRNAs_grouped = hmga1_hits[["name", "grouping"]].drop_duplicates().sort_values(by="grouping")

grouped_names_dict = (
    hmga1_snRNAs_grouped
    .groupby("grouping")["name"]
    .apply(list)
    .to_dict()
)

print(hmga1_snRNAs_grouped.to_string(index=False))

for group, names in grouped_names_dict.items():
    name_list_str = ", ".join(names)
    print(f"{group}: {name_list_str}")

for group, names in grouped_names_dict.items():
    short_names = [name.split('|')[0] for name in names]
    name_list_str = ", ".join(short_names)
    print(f"{group}: {name_list_str}")

############################################################################################

###################
## look as class ##
###################

# Step 1: Define canonical snRNA names explicitly
canonical_snrnas = [
    "RNU6ATAC", "U6", "RNU4ATAC", "RNU4-1", "RNU-1", "RNU2-1",
    "RNU5A-1", "RNU11", "RNU12", "RNU7-1"
]

# Step 2: Filter upstream scan results based on canonical snRNA names
merged_upstream_df_canonical = upstream_200_50_filtered_df_collapsed_df[
    upstream_200_50_filtered_df_collapsed_df["snrna_name"].isin(canonical_snrnas)
]

############################################################################################

# Get distinct TF families and snRNA groupings
canonical_families = (
    merged_upstream_df_canonical[["class", "grouping"]]
    .dropna()
    .drop_duplicates()
)

family_snRNA_counts = (
    upstream_200_50_filtered_df_collapsed_df
    .dropna(subset=["class"])
    .groupby(["class", "grouping"])["name"]
    .nunique()
    .reset_index(name="snRNA_count")
)

snRNA_by_types_paralog_counts = (
    upstream_bed[["name", "grouping"]]
    .drop_duplicates()
    .groupby("grouping")
    .size()
    .reset_index(name="unique_snRNA_count")
)

family_snRNA_counts_canonical_only = pd.merge(
    family_snRNA_counts,
    canonical_families,
    on=["class", "grouping"],
    how="inner"
)

normalized_family_counts = pd.merge(
    family_snRNA_counts_canonical_only,
    snRNA_by_types_paralog_counts,
    on="grouping",
    how="left"
)

# Normalize
normalized_family_counts["normalized_snRNA_count"] = (
    normalized_family_counts["snRNA_count"] / normalized_family_counts["unique_snRNA_count"]
)

normalized_family_counts["percent_snRNAs_with_family"] = (
    normalized_family_counts["normalized_snRNA_count"] * 100
).round(2)

group_order = ["U1", "U2", "U4", "U5", "U6", "U7", "U11", "U12", "U4ATAC", "U6ATAC", "Unknown"]

normalized_family_counts["grouping"] = pd.Categorical(
    normalized_family_counts["grouping"],
    categories=group_order,
    ordered=True
)

normalized_family_counts = normalized_family_counts.sort_values(
    by=["grouping", "percent_snRNAs_with_family"],
    ascending=[True, False]
).reset_index(drop=True)

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import seaborn as sns

# Create output directory
os.makedirs("canonical_motif_heatmaps", exist_ok=True)

# Clean and prepare columns
merged_upstream_df_canonical_motifs["grouping"] = merged_upstream_df_canonical_motifs["grouping"].str.strip().str.upper()
upstream_200_50_filtered_df_collapsed_df["grouping"] = upstream_200_50_filtered_df_collapsed_df["grouping"].str.strip().str.upper()
upstream_200_50_filtered_df_collapsed_df["motif_id"] = upstream_200_50_filtered_df_collapsed_df["motif_id"].astype(str).str.strip()
upstream_200_50_filtered_df_collapsed_df["motif_alt_id"] = upstream_200_50_filtered_df_collapsed_df["motif_alt_id"].astype(str).str.strip()

# Step 1: Canonical motif dict per group
canonical_alt_dict = (
    merged_upstream_df_canonical_motifs
    .groupby("grouping")["motif_alt_id"]
    .apply(set)
    .to_dict()
)

# Step 2: Input full motif table
df = upstream_200_50_filtered_df_collapsed_df.copy()
df["present"] = 1

# Step 3: Count canonical motif matches per group
match_counts = []
for group, canonical_set in canonical_alt_dict.items():
    count = df[(df["grouping"] == group) & (df["motif_alt_id"].isin(canonical_set))]["motif_id"].nunique()
    match_counts.append((group, count))
ranked_groups = [g for g, _ in sorted(match_counts, key=lambda x: x[1], reverse=True)]

# Step 4: Plot loop
## with only those with canonical motifs

for group in ranked_groups:
    canonical_alt_ids = canonical_alt_dict.get(group, set())
    df_group = df[df["grouping"] == group]
    df_group_canon = df_group[df_group["motif_alt_id"].isin(canonical_alt_ids)]

    if df_group_canon.empty:
        continue

    # Pivot: snRNA x motif_id
    all_snrnas = df_group["snrna_name"].dropna().unique()
    pivot_df = (
        df_group_canon.pivot_table(index="snrna_name", columns="motif_id", values="present", fill_value=0)
        .reindex(index=sorted(all_snrnas), fill_value=0)
    )

    # Map motif_id → motif_alt_id
    motif_id_to_alt = (
        df_group_canon[["motif_id", "motif_alt_id"]]
        .drop_duplicates()
        .groupby("motif_id")["motif_alt_id"]
        .agg(lambda x: "/".join(sorted(set(x))))
        .to_dict()
    )
    pivot_df.columns = [motif_id_to_alt.get(mid, mid) for mid in pivot_df.columns]
    pivot_df = pivot_df[sorted(pivot_df.columns)]
    # Sort snRNAs by motif count, keep only snRNAs with ≥1 motif
    pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]
    pivot_df = pivot_df[pivot_df.sum(axis=1) > 0]


    if pivot_df.shape[1] > 0:
        fig_width = min(max(6, 0.3 * pivot_df.shape[1]), 60)
        fig_height = min(max(2, 0.4 * pivot_df.shape[0]), 100)
        
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(pivot_df, cmap="Greys", cbar=False, linewidths=0.5, linecolor="lightgrey")
        plt.title(f"Canonical Motif Matches for {group} snRNAs")
        plt.xlabel("Motif (alt ID)")
        plt.ylabel("snRNA Name")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"canonical_motif_heatmaps/{group}_canonical_motifs.png", dpi=300)
        plt.close()

#################################################################################################################################

## with all snRNA
upstream_200_50_filtered_df_collapsed_df.to_csv("./upstream_200_50_motifs.csv", index=False)
merged_upstream_df_canonical_motifs
for group in ranked_groups:
    canonical_alt_ids = canonical_alt_dict.get(group, set())
    df_group = df[df["grouping"] == group]
    
    if df_group.empty:
        continue

    df_group_canon = df_group[df_group["motif_alt_id"].isin(canonical_alt_ids)]
    
    # Get all snRNAs present in this group
    all_snrnas = sorted(df_group["snrna_name"].dropna().unique())

    # Pivot only canonical hits, but include all snRNAs (fill missing rows with 0s)
    pivot_df = (
        df_group_canon.pivot_table(index="snrna_name", columns="motif_id", values="present", fill_value=0)
        .reindex(index=all_snrnas, fill_value=0)  # ensures snRNAs like RNU12 stay in
    )

    if pivot_df.empty or len(pivot_df.index) == 0:
        continue

    # Map motif_id → motif_alt_id (clean and readable column names)
    motif_id_to_alt = (
        df_group_canon[["motif_id", "motif_alt_id"]]
        .drop_duplicates()
        .groupby("motif_id")["motif_alt_id"]
        .agg(lambda x: "/".join(sorted(set(x))))
        .to_dict()
    )
    pivot_df.columns = [motif_id_to_alt.get(mid, mid) for mid in pivot_df.columns]
    pivot_df = pivot_df[sorted(pivot_df.columns)]

    # Sort snRNAs by canonical motif count (descending)
    pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]

    # Plot if there’s at least one snRNA
    fig_width = min(max(6, 0.3 * pivot_df.shape[1]), 60)
    fig_height = min(max(2, 0.4 * pivot_df.shape[0]), 100)

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(pivot_df, cmap="Greys", cbar=False, linewidths=0.5, linecolor="lightgrey")
    plt.title(f"Canonical Motif Matches for {group} snRNAs")
    plt.xlabel("Motif (alt ID)")
    plt.ylabel("snRNA Name")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"canonical_motif_heatmaps/{group}_canonical_motifs.png", dpi=300)
    plt.close()

#################################################################################################################################
## get class ####################################################################################################################
#################################################################################################################################

import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

matrix_ids = merged_upstream_df["motif_id"].drop_duplicates().tolist()

def get_class_and_family(matrix_id):
    url = f"https://jaspar.elixir.no/matrix/{matrix_id}"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        class_, family_ = None, None
        table = soup.find("table", id="matrix-detail")
        if table:
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) == 2:
                    label = cells[0].get_text(strip=True).rstrip(":").lower()
                    value = cells[1].get_text(strip=True)
                    if label == "class":
                        class_ = value
                    elif label == "family":
                        family_ = value

        return {"matrix_id": matrix_id, "class": class_, "family": family_}
    except Exception:
        return {"matrix_id": matrix_id, "class": None, "family": None}

# Parallelize the scraping with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=12) as executor:
    results = list(executor.map(get_class_and_family, matrix_ids))

jaspar_class_family_df = pd.DataFrame(results)
jaspar_class_family_df.to_csv("jaspar_class_family_df.csv")

#jaspar_class_family_df_test = pd.read_csv("jaspar_class_family_df.csv", index_col=0)
