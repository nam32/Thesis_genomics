"""

Purpose:
--------
Extract token-level activation scores from a fine-tuned DNABERT-2 model for a subset of sequences.
This version is designed to run on sequence slices (start:end), enabling parallelisation or job splitting.

Process:
--------
1. Loads a fine-tuned DNABERT-2 model and tokenizer from a local path.
2. Reads an input CSV file containing a 'sequence' column.
3. Processes only rows from start index to end index (exclusive).
4. For each sequence:
   - Tokenizes into model's BPE tokens (with CLS and SEP).
   - Runs the model to get hidden states and logits.
   - Computes the L2 norm of the last hidden layer embedding per token.
   - Records token string, token position, activation score, and predicted label.
5. Saves all token activations for the slice to a CSV file.

Arguments:
----------
--start        : Start index (inclusive) of sequences to process.
--end          : End index (exclusive) of sequences to process.
--input        : Path to CSV containing sequences (must have 'sequence' column).
--output       : Path to save token-level activation CSV.
--model_path   : Path to fine-tuned DNABERT-2 model directory.

Output CSV Columns:
-------------------
- sequence_id       : Original row index from input file.
- token             : Token string.
- token_position    : Position within the sequence (0-based, excluding CLS/SEP).
- activation_score  : L2 norm of token's embedding from last hidden layer.
- predicted_label   : Predicted class index for the sequence.
"""

import os
import sys
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, required=True)
parser.add_argument("--end", type=int, required=True)
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

#  Model + Tokenizer
sys.path.append(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_path, output_hidden_states=True, trust_remote_code=True
)
model.eval()

# Load Subset of Sequences
df = pd.read_csv(args.input)
sub_df = df.iloc[args.start:args.end]
assert "sequence" in sub_df.columns, "Missing 'sequence' column"

results = []

# Process Each Sequence
for idx, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
    seq_id = args.start + idx
    seq = row["sequence"]
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1].squeeze(0)
    activation_scores = last_hidden.norm(dim=-1).tolist()
    pred_label = outputs.logits.argmax(dim=-1).item()

    for i in range(1, len(tokens) - 1):  # skip [CLS] and [SEP]
        results.append({
            "sequence_id": seq_id,
            "token": tokens[i],
            "token_position": i - 1,
            "activation_score": activation_scores[i],
            "predicted_label": pred_label
        })

pd.DataFrame(results).to_csv(args.output, index=False)
print(f"\nSaved {len(results)} token activations to: {args.output}")
