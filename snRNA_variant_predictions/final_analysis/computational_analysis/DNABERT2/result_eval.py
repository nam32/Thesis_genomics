import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
This script merges DNABERT-2 prediction outputs with snRNA sequence annotations,
assigns each sequence to an snRNA class, and produces multiple summary tables
and plots. It generates:
1. Class-specific promoter probability dot plots (minor vs major snRNAs).
2. Predicted promoter strength distributions (count plots) for each snRNA class.
3. Violin plots showing model confidence for each predicted or true class.
4. Validation metrics (accuracy, F1, confusion matrix) and error analysis.
5. Separate Pol II prediction plots for snRNA classes.

The code handles both binary and multiclass promoter models, supports
probability-based ranking of predictions, and includes regex-based snRNA
class assignment to group related variants.
"""

################################################################################## Paths

csv_with_names = "/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results/rfam_ensembl/promoter_regions.csv"
predictions_path = "/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/dnabert2_custom_data_promoter_2_snRNA_predictions.csv"
output_path = "/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/dnabert2_custom_data_promoter_2_snRNA_predictions_with_headers.csv"

# #################################################################################
# Step 1: Load
df_names = pd.read_csv(csv_with_names)  # Expects 'name' and 'sequence' columns
df_preds = pd.read_csv(predictions_path)  # Expects 'sequence', 'prediction', 'label', etc.
df_names["sequence"] = df_names["sequence"].str.upper()
df_preds["sequence"] = df_preds["sequence"].str.upper()

#################################################################################
# # Step 3: Merge on sequence
merged = pd.merge(df_names, df_preds, on="sequence", how="inner")

# #################################################################################
# Step 4: Save to CSV
df_sorted = merged.sort_values(by="prediction")
df_sorted.to_csv(output_path, index=False)

#################################################################################
# Step 5: Results analysis

df_sorted[["name1", "name2"]] = df_sorted["name"].str.split("|", n=1, expand=True)

labels = ["U6atac", "U6", "U4atac", "U4", "U1", "U2", "U5", "U11", "U12", "U7"]
conditions = [
    df_sorted["name1"].str.contains(r"\bU6ATAC\b|^RNU6ATAC", case=False, na=False),
    df_sorted["name1"].str.contains(r"\bU6\b|^RNU6(?!ATAC)", case=False, na=False),
    df_sorted["name1"].str.contains(r"\bU4ATAC\b|^RNU4ATAC", case=False, na=False),
    df_sorted["name1"].str.contains(r"\bU4\b|^RNU4-", case=False, na=False),
    df_sorted["name1"].str.contains(r"\bU1\b|^RNU1-|^RNVU1", case=False, na=False),
    df_sorted["name1"].str.contains(r"\bU2\b|^RNU2", case=False, na=False),
    df_sorted["name1"].str.contains(r"\bU5\b|^RNU5", case=False, na=False),
    df_sorted["name1"].str.contains(r"\bU11\b|^RNU11", case=False, na=False),
    df_sorted["name1"].str.contains(r"\bU12\b|^RNU12", case=False, na=False),
    df_sorted["name1"].str.contains(r"\bU7\b|^RNU7", case=False, na=False),
]

df_sorted["snrna_class"] = np.select(conditions, labels, default="Other")
df_sorted = df_sorted.sort_values(by="snrna_class").reset_index(drop=True)

df_promoters = df_sorted[df_sorted['prediction'] == 1].reset_index(drop=True)
df_promoters = df_promoters.sort_values(by=["snrna_class", "prob_promoter"], ascending=[True, False]).reset_index(drop=True)
df_promoters = df_promoters.drop(columns=["sequence", "prediction", "prob_non-promoter", "name", "name2"])
df_promoters.to_csv("/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/snrna_promoters_predicted.csv", index=False)

df_promoters_minor = df_promoters[df_promoters['snrna_class'].isin(['U11', 'U12', 'U4atac', 'U6atac'])].reset_index(drop=True)

# Bar plot for promoter probabilities of minor spliceosome snRNA predictions

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(data=df_promoters_minor, x='snrna_class', y='prob_promoter', hue='snrna_class',
              palette='muted', jitter=True, size=8, ax=ax)

ax.axhline(0.5, color='grey', linestyle='--', label='Decision Threshold (0.5)')
ax.set_title('Predicted Promoter Probability by snRNA Class (Dot Plot)')
ax.set_ylabel('Predicted Probability (Promoter)')
ax.set_xlabel('snRNA Class')
ax.legend(title='snRNA Class', bbox_to_anchor=(1.05, 1), loc='upper left')

output_path = '/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/snRNA_promoter_probability_dotplot.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()

#################################################################################

fig, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(data=df_promoters_minor, x='snrna_class', y='prob_promoter', hue='snrna_class',
              palette='muted', jitter=True, size=8, ax=ax)

# Add upright, small, close labels
for i, row in df_promoters_minor.iterrows():
    ax.text(x=row['snrna_class'], y=row['prob_promoter'] + 0.005, s=row['name1'],
            ha='center', va='bottom', fontsize=6, rotation=0)

# Add threshold line
ax.axhline(0.5, color='grey', linestyle='--', label='Decision Threshold (0.5)')
ax.set_title('Predicted Promoter Probability by snRNA Class (Labeled Dots)')
ax.set_ylabel('Predicted Probability (Promoter)')
ax.set_xlabel('snRNA Class')
ax.legend(title='snRNA Class', bbox_to_anchor=(1.05, 1), loc='upper left')
output_path = '/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/snRNA_promoter_probability_dotplot.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

##################################################################################
##################################################################################
##################################################################################

dnabert2_df = pd.read_csv("/storage/homefs/tj23t050/BERT/190525/result/bert_snRNA_predictions_with_headers_2.csv")
dnabert2_df[["name1", "name2"]] = dnabert2_df["name"].str.split("|", n=1, expand=True)
dnabert2_df = dnabert2_df.drop(columns=["name", "sequence"])

dnabert2_df["label_prob"] = dnabert2_df.apply(
    lambda row: row[f"prob_{row['label']}"],
    axis=1
)

# Define snRNA gene class labels and regex conditions
labels = ["U6atac", "U6", "U4atac", "U4", "U1", "U2", "U5", "U11", "U12", "U7"]
conditions = [
    dnabert2_df["name1"].str.contains(r"\bU6ATAC\b|^RNU6ATAC", case=False, na=False),
    dnabert2_df["name1"].str.contains(r"\bU6\b|^RNU6(?!ATAC)", case=False, na=False),
    dnabert2_df["name1"].str.contains(r"\bU4ATAC\b|^RNU4ATAC", case=False, na=False),
    dnabert2_df["name1"].str.contains(r"\bU4\b|^RNU4-", case=False, na=False),
    dnabert2_df["name1"].str.contains(r"\bU1\b|^RNU1-|^RNVU1", case=False, na=False),
    dnabert2_df["name1"].str.contains(r"\bU2\b|^RNU2", case=False, na=False),
    dnabert2_df["name1"].str.contains(r"\bU5\b|^RNU5", case=False, na=False),
    dnabert2_df["name1"].str.contains(r"\bU11\b|^RNU11", case=False, na=False),
    dnabert2_df["name1"].str.contains(r"\bU12\b|^RNU12", case=False, na=False),
    dnabert2_df["name1"].str.contains(r"\bU7\b|^RNU7", case=False, na=False),
]

# Apply conditions to assign snRNA class
dnabert2_df["snrna_class"] = np.select(conditions, labels, default="Other")

################################## PLOT
##################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the DataFrame (simulate input)
df = dnabert2_df.copy()

# Only keep specific snRNA classes
snrna_subset = ["U1", "U2", "U4", "U6"]
df_filtered = df[df["snrna_class"].isin(snrna_subset)]

# Set plot style
sns.set(style="whitegrid", font_scale=1.1)

# Plot
plt.close("all")
g = sns.catplot(
    data=df_filtered,
    x="label",
    col="snrna_class",
    col_order=["U1", "U2", "U4", "U6"],
    kind="count",
    col_wrap=4,
    order=["non-promoter", "weak", "strong"],
    color="#5AB1D3",
    height=4,
    aspect=0.9
)

axes = g.axes.flatten()
for ax in axes:
    ax.set_ylim(0, 600)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor("black")

g.set_titles("{col_name}")

g.fig.subplots_adjust(top=0.80)
g.fig.suptitle("Predicted Promoter Strength by snRNA Class")
g.set_axis_labels("Predicted Class", "Count")

output_path = "/storage/homefs/tj23t050/BERT/190525/result/dnabert2_class_distribution_major.png"
plt.savefig(output_path, bbox_inches="tight", dpi=300)
plt.show()

#################################################################################

# Only keep specific snRNA classes
snrna_subset = ["U11", "U12", "U4atac", "U6atac"]
df_filtered = df[df["snrna_class"].isin(snrna_subset)]

# Set plot style
sns.set(style="whitegrid", font_scale=1.1)

# Plot
plt.close("all")
g = sns.catplot(
    data=df_filtered,
    x="label",
    col="snrna_class",
    col_order=["U11", "U12", "U4atac", "U6atac"],
    kind="count",
    col_wrap=4,
    order=["non-promoter", "weak", "strong"],
    color="#638AC7",
    height=4,
    aspect=0.9
)

axes = g.axes.flatten()
for ax in axes:
    ax.set_ylim(0, 25)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor("black")

# Remove "snrna_class = " prefix from subplot titles
g.set_titles("{col_name}")

# Style and save
g.fig.subplots_adjust(top=0.80)
g.set_axis_labels("Predicted Class", "Count")

output_path = "/storage/homefs/tj23t050/BERT/190525/result/dnabert2_class_distribution_minor.png"
plt.savefig(output_path, bbox_inches="tight", dpi=300)
plt.show()

#################################################################################

import seaborn as sns
import matplotlib.pyplot as plt

label_map = {0: "non-promoter", 1: "weak", 2: "strong"}
dnabert2_df["predicted_label"] = dnabert2_df["prediction"].map(label_map)

label_order = ["non-promoter", "weak", "strong"]

# Set style
sns.set(style="whitegrid", font_scale=1.2)
palette = {
    "non-promoter": "#4C72B0",
    "weak": "#55A868",
    "strong": "#C44E52"
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Violin plot without inner box
sns.violinplot(
    x="predicted_label", y="label_prob", data=dnabert2_df,
    palette=palette, inner=None, ax=ax, order=label_order
)

# Swarmplot to show individual points
sns.swarmplot(
    x="predicted_label", y="label_prob", data=dnabert2_df,
    color="black", size=2, alpha=0.4, ax=ax, order=label_order
)

# horizontal lines for medians
for i, label in enumerate(label_order):
    mean = dnabert2_df.loc[dnabert2_df["predicted_label"] == label, "label_prob"].mean()
    ax.hlines(y=mean, xmin=i - 0.4, xmax=i + 0.4, color="darkred", linestyle="--", linewidth=2)
    ax.text(i, mean + 0.015, f"Mean: {mean:.2f}", ha='center', va='bottom', fontsize=9, color='darkred')

ax.set_ylabel("Probability Assigned to True Label")
ax.set_xlabel("Predicted Class")
ax.set_ylim(0, 1)
plt.tight_layout()
output_path = "/storage/homefs/tj23t050/BERT/190525/result/dnabert2_violin_plot_test_2.png"
plt.savefig(output_path, bbox_inches="tight")
plt.show()

##########################################################
################### Validation set #######################
##########################################################

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the two input files
prediction_df = pd.read_csv("/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/dnabert2_custom_data_promoter_2_val_predictions.csv")
promoter_df = pd.read_csv("/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/final_training_data/human_tf_all_test.csv")

val_df = pd.merge(promoter_df, prediction_df, on="sequence", how="inner")

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Prep dataframe
val_df = val_df.rename(columns={"label_y": "pred", "prediction": "binary_pred", "label_x": "groundtruth_binary"})

# Accuracy: 0.7705
accuracy = accuracy_score(val_df["groundtruth_binary"], val_df["binary_pred"])
print(f"Accuracy: {accuracy:.4f}")

# F1 Score: 0.7377
f1 = f1_score(val_df["groundtruth_binary"], val_df["binary_pred"])
print(f"F1 Score: {f1:.4f}")

# Compute confusion matrix
cm = confusion_matrix(val_df["groundtruth_binary"], val_df["binary_pred"])

# Display as heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-promoter", "Promoter"])
disp.plot(cmap="Blues", values_format="d")

plt.title("Confusion Matrix - DNABERT-2 Predictions")
plt.savefig("/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/confusion_matrix_dnabert2_custom_data_promoter_2.png", dpi=300, bbox_inches='tight')  # You can change filename and format

plt.show()

################################################
################Check for only wrong pred#######
################################################

wrong_pred_df = val_df[val_df['groundtruth_binary'] != val_df['binary_pred']]
wrong_pred_df = wrong_pred_df.copy()

files = [
    "/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/promoter_tata_polII.tsv",
    "/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/promoters_non_tata_polII.tsv",
    "/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/non_polII_promoters.tsv",
    "/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/non_promoter_tata_like.tsv",
    "/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/non_promoter_non_tata_sequences.tsv"
]

file_sequences = {}
for file in files:
    with open(file, 'r') as f:
        sequences = set(line.strip().split(',')[0] for line in f if line.strip())
        file_sequences[file] = sequences


def find_source_file(seq):
    for fname, seq_set in file_sequences.items():
        if seq in seq_set:
            return fname
    return None
import os

wrong_pred_df['source_file'] = wrong_pred_df['sequence'].apply(find_source_file)
wrong_pred_df['source_label'] = wrong_pred_df['source_file'].apply(
    lambda x: os.path.splitext(os.path.basename(x))[0] if pd.notnull(x) else None
)

source_counts = wrong_pred_df['source_label'].value_counts()

############################
## Plot

plot_df = val_df[['groundtruth_binary', 'prob_promoter', "binary_pred"]].copy()
plot_df['Class'] = plot_df['groundtruth_binary'].map({0: 'Non-Promoter', 1: 'Promoter'})

# Add predicted class labels based on binary_pred
plot_df['Predicted Class'] = plot_df['binary_pred'].map({0: 'Non-Promoter', 1: 'Promoter'})
plot_df['True Class'] = plot_df['groundtruth_binary'].map({0: 'Non-Promoter', 1: 'Promoter'})

fig, ax = plt.subplots(figsize=(8, 6))

# Split violin plot for distribution separation by True Class
sns.violinplot(y='Predicted Class', x='prob_promoter', data=plot_df, hue='True Class',
               palette={"Non-Promoter": "lightcoral", "Promoter": "lightblue"}, split=True, inner=None, ax=ax)

# Overlay dots colored by true class
sns.stripplot(y='Predicted Class', x='prob_promoter', data=plot_df, hue='True Class',
              palette={"Non-Promoter": "darkred", "Promoter": "darkblue"}, dodge=True, size=2, alpha=0.5, ax=ax)

# Decision threshold line
ax.axvline(0.5, color='grey', linestyle='--')

ax.set_title('Predicted Promoter Probability by Predicted Class (Split Violin + Colored Dots)')
ax.set_xlabel('Predicted Probability (Promoter)')

# Clean legend to avoid duplicate entries
handles, labels = ax.get_legend_handles_labels()
# Only keep first two unique labels (Non-Promoter, Promoter)
ax.legend(handles[:2], labels[:2], title='True Class', bbox_to_anchor=(1.05, 1), loc='upper left')
output_path = "/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/predicted_class_violin_plot.png"
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

###########################
### PolII and non PolII ###
###########################
# ───────────────────────────────────────────────────────────────────────────────
# Paths
csv_with_names = "/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results/rfam_ensembl/promoter_regions.csv"
polII_path = "/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/dnabert2_custom_data_polII_1.csv"
output_path = "/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/dnabert2_custom_data_polII_1_snRNA_predictions_with_headers.csv"

# ───────────────────────────────────────────────────────────────────────────────
# Step 1: Load
df_names = pd.read_csv(csv_with_names)
df_polII = pd.read_csv(polII_path)
df_names["sequence"] = df_names["sequence"].str.upper()
df_polII["sequence"] = df_polII["sequence"].str.upper()

# ───────────────────────────────────────────────────────────────────────────────
# Step 3: Merge on sequence
merged_polII = pd.merge(df_names, df_polII, on="sequence", how="inner")

merged_polII[["name1", "name2"]] = merged_polII["name"].str.split("|", n=1, expand=True)

labels = ["U6atac", "U6", "U4atac", "U4", "U1", "U2", "U5", "U11", "U12", "U7"]
conditions = [
    merged_polII["name1"].str.contains(r"\bU6ATAC\b|^RNU6ATAC", case=False, na=False),
    merged_polII["name1"].str.contains(r"\bU6\b|^RNU6(?!ATAC)", case=False, na=False),
    merged_polII["name1"].str.contains(r"\bU4ATAC\b|^RNU4ATAC", case=False, na=False),
    merged_polII["name1"].str.contains(r"\bU4\b|^RNU4-", case=False, na=False),
    merged_polII["name1"].str.contains(r"\bU1\b|^RNU1-|^RNVU1", case=False, na=False),
    merged_polII["name1"].str.contains(r"\bU2\b|^RNU2", case=False, na=False),
    merged_polII["name1"].str.contains(r"\bU5\b|^RNU5", case=False, na=False),
    merged_polII["name1"].str.contains(r"\bU11\b|^RNU11", case=False, na=False),
    merged_polII["name1"].str.contains(r"\bU12\b|^RNU12", case=False, na=False),
    merged_polII["name1"].str.contains(r"\bU7\b|^RNU7", case=False, na=False),
]

merged_polII["snrna_class"] = np.select(conditions, labels, default="Other")
merged_polII = merged_polII.sort_values(by="snrna_class").reset_index(drop=True)
merged_polII = merged_polII.drop(columns=["sequence", "prediction", "name", "name2"])

#merged_polII = merged_polII.sort_values(by=["snrna_class", "prob_promoter"], ascending=[True, False]).reset_index(drop=True)
#df_promoters = df_promoters.drop(columns=["sequence", "prediction", "prob_non-promoter", "name", "name2"])
#df_promoters.to_csv("/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/snrna_promoters_predicted.csv", index=False)

df_polII_minor = merged_polII[merged_polII['snrna_class'].isin(['U11', 'U12', 'U4atac', 'U6atac'])].reset_index(drop=True)

##################################################################################

# Step 4: Save to CSV
merged_polII = merged_polII.sort_values(by="prediction")
merged_polII.to_csv(output_path, index=False)


fig, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(data=merged_polII, x='snrna_class', y='prob_polII', hue='label',
              palette={"polII": "blue", "non-polII": "red"}, jitter=True, size=8, ax=ax)

ax.axhline(0.5, color='grey', linestyle='--', label='Decision Threshold (0.5)')
ax.set_title('Predicted Pol II Probability by snRNA Class (Coloured by Predicted Class)')
ax.set_ylabel('Predicted Probability (Pol II)')
ax.set_xlabel('snRNA Class')
ax.legend(title='Predicted Class', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/polII_pred.png", dpi=300, bbox_inches='tight')  # You can change filename and format

plt.show()

###

fig, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(data=merged_polII, x='snrna_class', y='prob_polII', hue='label',
              palette={"polII": "blue", "non-polII": "red"}, jitter=True, size=8, alpha=0.4, ax=ax)

ax.axhline(0.5, color='grey', linestyle='--', label='Decision Threshold (0.5)')
ax.set_title('Predicted Pol II Probability by snRNA Class (Transparent Circles)')
ax.set_ylabel('Predicted Probability (Pol II)')
ax.set_xlabel('snRNA Class')
ax.legend(title='Predicted Class', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig("/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/polII_pred.png", dpi=300, bbox_inches='tight')  # You can change filename and format
