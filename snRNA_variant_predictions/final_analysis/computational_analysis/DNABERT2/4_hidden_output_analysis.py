"""

Purpose
-------
Script to extract the top-N tokens per predicted class based on average activation scores.

This script:
1. Loads a CSV containing tokens, predicted labels, and activation scores.
2. Removes padding tokens ([PAD]) that do not carry semantic meaning.
3. Groups tokens by predicted class and computes the mean activation score for each unique token.
4. Selects the top-N tokens with the highest activation scores per class.
5. Saves the results to a new CSV file for downstream analysis or visualisation.

Expected CSV columns:
- token: the token string (e.g., "ATG", "promoter", etc.)
- predicted_label: integer or string class label predicted by the model
- activation_score: numeric score representing token activation
"""

import pandas as pd

df = pd.read_csv("hidden_output_all.csv")

# Remove padding tokens
df = df[df["token"] != "[PAD]"]

# Set top N tokens per class
top_n = 21

top_tokens_all = []

# Loop over predicted classes
for label, group in df.groupby("predicted_label"):
    # Average activation score per unique token
    top_tokens = (
        group.groupby("token", as_index=False)
             .agg({"activation_score": "mean"})
             .sort_values("activation_score", ascending=False)
             .head(top_n)
    )
    top_tokens["label"] = label
    top_tokens_all.append(top_tokens)

# Concatenate results
if top_tokens_all:
    result = pd.concat(top_tokens_all)
    result = result[["label", "token", "activation_score"]]
    result.to_csv("top_tokens_by_class.csv", index=False)
    print("✅ Saved top tokens to: top_tokens_by_class.csv")
else:
    print("⚠️ No valid tokens found — check your input.")
