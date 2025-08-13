"""
Script: get_all_token_activations.py

Purpose:
--------
Extracts per-token activation scores from a fine-tuned DNABERT-2 model for a set of input DNA sequences.

Process:
--------
1. Loads a fine-tuned DNABERT-2 model and its tokenizer from a local path.
2. Reads an input CSV file containing a 'sequence' column.
3. For each sequence:
   - Tokenizes into DNABERT-2's subword tokens (BPE).
   - Runs the model to obtain hidden states for all tokens.
   - Computes the L2 norm of the last hidden layer embedding for each token.
   - Records token string, position, activation score, and predicted label.
4. Saves all per-token activation scores into a CSV for downstream analysis.

Inputs:
-------
- MODEL_PATH   : Path to the fine-tuned DNABERT-2 model directory.
- INPUT_FILE   : CSV containing DNA sequences in a column named 'sequence'.
- SAVE_OUTPUT  : Output CSV to store per-token activations.

Output CSV Columns:
-------------------
- sequence_id       : Index of the sequence in the input CSV.
- token             : Token string from tokenizer.
- token_position    : Position in the sequence (0-based, excluding CLS/SEP).
- activation_score  : L2 norm of the token's last-layer embedding.
- predicted_label   : Model's predicted class for the sequence.

Note:
-----
- Skips [CLS] and [SEP] tokens, as they are special tokens not part of the DNA sequence.
- Uses `output_hidden_states=True` to retrieve intermediate model representations.
"""

import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "/storage/homefs/tj23t050/BERT/dnabert2_custom_data_promoter_2"
INPUT_FILE = "/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/final_training_data/human_tf_all_train.csv"
SAVE_OUTPUT = "/storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/hidden_output_all_tokens.csv"

# Add local DNABERT-2 module path
sys.path.append(MODEL_PATH)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    output_hidden_states=True,
    trust_remote_code=True
)
model.eval()

# Load input sequences
df = pd.read_csv(INPUT_FILE)
assert "sequence" in df.columns, "CSV must contain a 'sequence' column."

results = []

# Loop over sequences
for idx, row in tqdm(df.iterrows(), total=len(df)):
    seq = row["sequence"]

    # Tokenise
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Run model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # Tuple of layers
    last_hidden = hidden_states[-1].squeeze(0)  # shape: (seq_len, hidden_dim)
    logits = outputs.logits
    pred_label = logits.argmax(dim=-1).item()

    # Compute L2 norm (activation score) for each token
    activation_scores = last_hidden.norm(dim=-1).tolist()

    for i in range(1, len(tokens) - 1):  # skip [CLS] and [SEP]
        token = tokens[i]
        score = activation_scores[i]
        results.append({
            "sequence_id": idx,
            "token": token,
            "token_position": i - 1,
            "activation_score": score,
            "predicted_label": pred_label
        })

# Save all token activations
pd.DataFrame(results).to_csv(SAVE_OUTPUT, index=False)
print(f"\n✅ Done. Full token activations saved to: {SAVE_OUTPUT}")
