"""

Purpose:
--------
Evaluates binary classification results from DNABERT-2, producing:
1. Accuracy score
2. Confusion matrix plot
3. Confidence violin plots:
   - Grouped by predicted class (with precision annotations)
   - Grouped by true class (with recall annotations)

Inputs:
-------
- prediction_df: CSV with model predictions and probabilities
- promoter_df:   CSV with ground truth labels and sequences

Merged on the 'sequence' column.

Expected columns after merging and renaming:
- groundtruth_binary : Ground truth class (0 = non-promoter, 1 = promoter)
- binary_pred        : Model-predicted class
- prob_non-promoter  : Model probability for non-promoter class
- prob_promoter      : Model probability for promoter class

Outputs:
--------
- Confusion matrix PNG
- Violin + swarm plots (by predicted class and by true class) with precision/recall annotations
"""

import pandas as pd
import numpy as np
import seaborn as sns
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

# multiclasses
# Accuracy: 0.7705
accuracy = accuracy_score(val_df["groundtruth_binary"], val_df["binary_pred"])
print(f"Accuracy: {accuracy:.4f}")

# Compute confusion matrix
cm = confusion_matrix(val_df["groundtruth_binary"], val_df["binary_pred"])

# Display as heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-promoter", "Promoter"])
disp.plot(cmap="Blues", values_format="d")

plt.title("Confusion Matrix - DNABERT-2 Predictions")
plt.savefig("/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/confusion_matrix_dnabert2_custom_data_promoter_2.png", dpi=300, bbox_inches='tight') 

plt.show()

########################################################################################
## Plot

label_map = {0: "non-promoter", 1: "promoter"}

# Map predicted and true labels
val_df["predicted_label"] = val_df["binary_pred"].map(label_map)
val_df["true_label"] = val_df["groundtruth_binary"].map(label_map)

# Probability assigned to true label
def get_label_prob(row):
    if row["groundtruth_binary"] == 0:
        return row["prob_non-promoter"]
    elif row["groundtruth_binary"] == 1:
        return row["prob_promoter"]
    return None

val_df["label_prob"] = val_df.apply(get_label_prob, axis=1)

## confidence per predicted class

import seaborn as sns
import matplotlib.pyplot as plt

palette = {"non-promoter": "#3498db", "promoter": "#e74c3c"}
label_order = ["non-promoter", "promoter"]

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Violin plot
sns.violinplot(
    x="predicted_label", y="label_prob", data=val_df,
    palette=palette, inner=None, ax=ax, order=label_order
)

# Swarmplot
sns.swarmplot(
    x="predicted_label", y="label_prob", data=val_df,
    hue="true_label", palette=palette, size=4, alpha=0.7,
    linewidth=0.6, edgecolor="black", ax=ax, order=label_order
)

# % Correct per predicted class
annotation_texts = []
for i, label in enumerate(label_order):
    subset = val_df[val_df["predicted_label"] == label]
    if not subset.empty:
        correct = (subset["predicted_label"] == subset["true_label"]).sum()
        percent_correct = 100 * correct / len(subset)
        annotation_texts.append((i, percent_correct))

# Mean lines + annotations
for i, label in enumerate(label_order):
    mean = val_df.loc[val_df["predicted_label"] == label, "label_prob"].mean()
    ax.hlines(y=mean, xmin=i - 0.4, xmax=i + 0.4, color="black", linestyle="--", linewidth=2)
    ax.text(i, mean + 0.02, f"Mean: {mean:.2f}", ha='center', va='bottom', fontsize=9, weight='bold')

# Precision annotations
for i, pct in annotation_texts:
    ax.text(i, -0.05, f"{pct:.1f}% precision", ha='center', va='top', fontsize=10, weight='bold')

ax.set_title("Model Confidence per Predicted Class (Binary)", pad=15)
ax.set_ylabel("Probability Assigned to True Label")
ax.set_xlabel("Predicted Class")
ax.set_ylim(-0.1, 1)

# Legend cleanup
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), title="True Class", loc='upper right')

plt.tight_layout()
plt.savefig("/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/dnabert2_violin_plot_binary_predicted.png", bbox_inches="tight", dpi=600)
plt.show()

########################################################################################

true_label_order = ["non-promoter", "promoter"]

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

sns.violinplot(
    x="true_label", y="label_prob", data=val_df,
    palette=palette, inner=None, ax=ax, order=true_label_order
)

sns.swarmplot(
    x="true_label", y="label_prob", data=val_df,
    hue="predicted_label", palette=palette, size=4, alpha=0.7,
    linewidth=0.6, edgecolor="black", ax=ax, order=true_label_order
)

# Recall annotations
for i, label in enumerate(true_label_order):
    mean = val_df.loc[val_df["true_label"] == label, "label_prob"].mean()
    ax.hlines(y=mean, xmin=i - 0.4, xmax=i + 0.4, color="black", linestyle="--", linewidth=2)
    ax.text(i, mean + 0.02, f"Mean: {mean:.2f}", ha='center', va='bottom', fontsize=9, weight='bold')

for i, label in enumerate(true_label_order):
    subset = val_df[val_df["true_label"] == label]
    if not subset.empty:
        correct = (subset["true_label"] == subset["predicted_label"]).sum()
        percent_correct = 100 * correct / len(subset)
        ax.text(i, -0.05, f"{percent_correct:.1f}% recall", ha='center', va='top', fontsize=10, weight='bold')

ax.set_title("Model Confidence Grouped by True Class (Binary)", pad=15)
ax.set_ylabel("Probability Assigned to True Label")
ax.set_xlabel("True Class")
ax.set_ylim(-0.1, 1)

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), title="Predicted Class", loc='upper right')

plt.tight_layout()
plt.savefig("/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/dnabert2_violin_plot_binary_trueclass.png", bbox_inches="tight", dpi=600)
plt.show()
