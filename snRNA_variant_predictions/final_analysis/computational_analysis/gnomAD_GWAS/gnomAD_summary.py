"""

Purpose:
-------
1) Load snRNA genomic intervals (GRCh38.p14) and per-variant population stats
   from an Excel workbook, then find all variants that overlap each snRNA gene.
2) For each snRNA gene, select the single top variant by global allele frequency (AF).
3) Save a summary CSV of these top variants per gene and generate figures:
   - Bar charts of top-variant AFs split into pseudogenes (…P) vs gene copies (RNU… not ending in P),
     coloured by inferred snRNA class (U1/U2/U4/U5/U6/U11/U12/U4ATAC/U6ATAC).
   - Stacked bar of within-gene population composition (AN proportions).
   - Multi-population bar chart of AFs with per-bar AN labels for the top genes.

Key I/O
------
Input:
- combined_variant_stats.xlsx
  - Sheet 'snRNA_GRCh38.p14' : reference snRNA coordinates and names
  - Sheet 'combined_variant_stats' : variant AF/AN by population (+ rsID, POS, CHROM, etc.)

Outputs:
- top_snRNA_variants.csv                  # top AF variant per gene
- A_within_each_gene.svg                  # stacked AN composition per gene
- population_AF_with_AN_labels.png        # AF per population with AN annotations

Assumptions & Notes
-------------------
- Coordinates are treated as inclusive and compared in the same build (GRCh38);
  both tables are normalised to chromosome names *without* the 'chr' prefix.
- Population columns are expected as AF_* and AN_* for AFR/AMI/AMR/ASJ/EAS/FIN/NFE/SAS.
"""

import pandas as pd

file_path = "C:/Users/tjanj/OneDrive/Desktop/Unibe/rubin_lab/combined_variant_stats.xlsx"
xls = pd.ExcelFile(file_path)


# Load
snrna_locations_df = pd.read_excel(xls, sheet_name='snRNA_GRCh38.p14')
variant_stats_df = pd.read_excel(xls, sheet_name='combined_variant_stats')

variant_stats_df["CHROM_clean"] = variant_stats_df["CHROM"].str.replace("chr", "", regex=False)
snrna_locations_df["Chromosome_clean"] = snrna_locations_df["Chromosome/scaffold name"].astype(str).str.replace("chr", "", regex=False)

matched_variants = []

for _, gene_row in snrna_locations_df.iterrows():
    chrom = gene_row["Chromosome_clean"]
    start = gene_row["Gene start (bp)"]
    end = gene_row["Gene end (bp)"]
    gene_name = gene_row["Gene name"]

    overlapping_variants = variant_stats_df[
        (variant_stats_df["CHROM_clean"] == chrom) &
        (variant_stats_df["POS"] >= start) &
        (variant_stats_df["POS"] <= end)
    ].copy()

    overlapping_variants["Matched_Gene"] = gene_name
    matched_variants.append(overlapping_variants)

matched_variants_df = pd.concat(matched_variants, ignore_index=True)

top_variants_per_gene = matched_variants_df.loc[
    matched_variants_df.groupby('Matched_Gene')['AF'].idxmax()
][['Matched_Gene', 'Paralog', 'rsID', 'AF', 'AN', 'CHROM', 'POS', 'AF_afr', 'AF_ami', 'AF_amr', 'AF_asj', 'AF_eas', 'AF_fin', 'AF_nfe', 'AF_sas', 'AN_afr', 'AN_ami', 'AN_amr', 'AN_asj', 'AN_eas', 'AN_fin', 'AN_nfe', 'AN_sas']].sort_values(by='AF', ascending=False)

print(top_variants_per_gene.head(10))
top_variants_per_gene.to_csv("top_snRNA_variants.csv", index=False) 

##########
## Plot ##
##########

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("top_snRNA_variants.csv")

# Classify snRNA type for coloring
def get_snRNA_type(gene_name):
    if "U6ATAC" in gene_name:
        return "U6ATAC"
    elif "U4ATAC" in gene_name:
        return "U4ATAC"
    elif "U11" in gene_name:
        return "U11"
    elif "U12" in gene_name:
        return "U12"
    elif "U1" in gene_name:
        return "U1"
    elif "U2" in gene_name:
        return "U2"
    elif "U4" in gene_name:
        return "U4"
    elif "U5" in gene_name:
        return "U5"
    elif "U6" in gene_name:
        return "U6"
    else:
        return "Other"

df['snRNA_Type'] = df['Matched_Gene'].apply(get_snRNA_type)

# Filter: Pseudogenes and Gene Copies only
df_pseudo = df[df['Matched_Gene'].str.endswith("P")]
df_copies = df[df['Matched_Gene'].str.startswith("RNU") & ~df['Matched_Gene'].str.endswith("P")]

# Color palette
unique_types = df['snRNA_Type'].unique()
palette = dict(zip(unique_types, sns.color_palette("tab10", len(unique_types))))

# Plot setup
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Plot 1: Pseudogenes
sns.barplot(x='Matched_Gene', y='AF', data=df_pseudo.sort_values(by='AF', ascending=False),
            ax=axes[0], hue='snRNA_Type', dodge=False, palette=palette)
axes[0].set_title("Top Variant AF in snRNA Pseudogenes")
axes[0].set_xlabel("Pseudogene")
axes[0].set_ylabel("Allele Frequency")
axes[0].tick_params(axis='x', rotation=90)
axes[0].legend(title="snRNA Type")

# Plot 2: Gene Copies
sns.barplot(x='Matched_Gene', y='AF', data=df_copies.sort_values(by='AF', ascending=False),
            ax=axes[1], hue='snRNA_Type', dodge=False, palette=palette)
axes[1].set_title("Top Variant AF in snRNA Gene Copies")
axes[1].set_xlabel("Gene Copy")
axes[1].set_ylabel("Allele Frequency")
axes[1].tick_params(axis='x', rotation=90)
axes[1].legend(title="snRNA Type")

plt.tight_layout()
plt.show()

#############################
## sub population analysis ##
#############################

import pandas as pd
import matplotlib.pyplot as plt

# Population full names
pop_full_names = {
    'AFR': 'African/African American',
    'AMI': 'Amish',
    'AMR': 'Admixed American',
    'ASJ': 'Ashkenazi Jewish',
    'EAS': 'East Asian',
    'FIN': 'Finnish',
    'NFE': 'Non-Finnish European',
    'SAS': 'South Asian'
}

# Fixed color palette for each population
pop_colors = {
    'African/African American': '#1f77b4',
    'Amish': '#ff7f0e',
    'Admixed American': '#2ca02c',
    'Ashkenazi Jewish': '#d62728',
    'East Asian': '#9467bd',
    'Finnish': '#8c564b',
    'Non-Finnish European': '#e377c2',
    'South Asian': '#7f7f7f'
}

df = pd.read_csv("top_snRNA_variants.csv")

# Population-specific AN columns (ensure alignment with keys above)
population_columns = ['AN_afr', 'AN_ami', 'AN_amr', 'AN_asj', 'AN_eas', 'AN_fin', 'AN_nfe', 'AN_sas']

# Group by gene and sum AN per population
gene_population_AN = df.groupby('Matched_Gene')[population_columns].sum()

# Normalize AN values within each gene (row-wise normalization)
within_gene_normalized_AN = gene_population_AN.div(gene_population_AN.sum(axis=1), axis=0)

full_column_names = [pop_full_names[col.split('_')[1].upper()] for col in population_columns]
within_gene_normalized_AN.columns = full_column_names

# Reorder columns to match pop_colors order
within_gene_normalized_AN = within_gene_normalized_AN[pop_colors.keys()]

# Plot with specified colors
within_gene_normalized_AN.plot(
    kind='bar', stacked=True, figsize=(14, 7),
    color=[pop_colors[pop] for pop in within_gene_normalized_AN.columns]
)

plt.ylabel('Proportion of AN per Gene')
plt.title('Population Proportion of Sample Size (AN) Within Each Gene')
plt.xticks(rotation=90, ha='right')
plt.legend(title='Population', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("A_within_each_gene.svg", format='svg')
plt.show()

#############################
## sub population analysis ##
#############################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("top_snRNA_variants.csv")

# Mapping from population code to full name (from your image)
pop_full_names = {
    'AFR': 'African/African American',
    'AMI': 'Amish',
    'AMR': 'Admixed American',
    'ASJ': 'Ashkenazi Jewish',
    'EAS': 'East Asian',
    'FIN': 'Finnish',
    'NFE': 'Non-Finnish European',
    'SAS': 'South Asian'
    #'OTH': 'Other'
}

# Define population labels and column names
pop_labels = ['AFR', 'AMI', 'AMR', 'ASJ', 'EAS', 'FIN', 'NFE', 'SAS']
af_cols = [f'AF_{p.lower()}' for p in pop_labels]
an_cols = [f'AN_{p.lower()}' for p in pop_labels]

# Convert AF and AN columns to numeric to ensure valid operations
for col in af_cols + an_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Prepare data for plotting (long format)
af_an_df = df[['Matched_Gene'] + af_cols + an_cols].copy()
melted_af_an = []

for idx, row in af_an_df.iterrows():
    for pop in pop_labels:
        melted_af_an.append({
            'Matched_Gene': row['Matched_Gene'],
            'Population': pop_full_names[pop],  # Use full name here
            'AF': row[f'AF_{pop.lower()}'],
            'AN': int(row[f'AN_{pop.lower()}']) if not pd.isna(row[f'AN_{pop.lower()}']) else 0
        })


plot_df = pd.DataFrame(melted_af_an)

# Set the order for legend and colors
population_order = list(pop_colors.keys())
palette = pop_colors

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure numeric columns
plot_df['AF'] = pd.to_numeric(plot_df['AF'], errors='coerce')
plot_df['AN'] = pd.to_numeric(plot_df['AN'], errors='coerce')

# Get unique populations from data
population_order = sorted(plot_df['Population'].unique())

# Create color palette
palette = dict(zip(population_order, sns.color_palette("tab10", len(population_order))))

# Get top 35 genes by overall AF
top_genes = plot_df.groupby('Matched_Gene')['AF'].max().nlargest(35).index

# Filter plot_df to only include those genes
plot_df_top = plot_df[plot_df['Matched_Gene'].isin(top_genes)]

# Plot
plt.figure(figsize=(24, 10))
ax = sns.barplot(
    data=plot_df_top,
    x='Matched_Gene',
    y='AF',
    hue='Population',
    hue_order=population_order,
    palette=palette
)

# Annotate AN on top of each bar
for container, pop in zip(ax.containers, population_order):
    sub_df = plot_df[plot_df['Population'] == pop]
    for bar, (_, row) in zip(container, sub_df.iterrows()):
        an = row['AN']
        if not pd.isna(an) and an > 0:
            height = bar.get_height()
            y = height if height > 0 else 0.01
            ax.annotate(f'{int(an)}',
                        (bar.get_x() + bar.get_width() / 2, y),
                        ha='center', va='bottom', fontsize=7, rotation=90)


plt.title("Allele Frequency per Population for Top Variants per snRNA Gene (with Sample Size)")
plt.ylabel("Allele Frequency (AF)")
plt.xlabel("snRNA Gene")
plt.xticks(rotation=90)
plt.legend(title='Population', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

plt.savefig("population_AF_with_AN_labels.png", format='png', dpi=300)
plt.show()

