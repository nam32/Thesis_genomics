import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import os
from datetime import datetime

"""

Purpose:
-------
Run inference with a fine-tuned DNABERT-2 classifier on a CSV of DNA sequences and
write predictions + class probabilities to disk.

Key points
----------
- DNABERT-2 uses BPE (byte-pair encoding). **Do not** pre-slice sequences into k-mers.
- Uses batched inference (GPU if available).
- Saves predictions, probabilities, and a small inference metadata file.

Inputs
------
- MODEL_DIR   : path to fine-tuned DNABERT-2 model folder
- INPUT_CSV   : CSV with a 'sequence' column
- OUTPUT_CSV  : path for saving predictions

Outputs
-------
- OUTPUT_CSV with columns:
  [sequence, prediction, label, prob_no, prob_yes, ... (plus original columns)]
- training_info.txt in MODEL_DIR with run metadata
"""

# All models:
# dnabert2_multiclass_model
# dnabert2_human_tf_all_model
# dnabert2_prom_300_all_model
# dnabert2_prom_300_tata_model
# dnabert2_prom_core_all_model
# dnabert2_prom_core_tata_model

# dnabert2_multiclass_model
#MODEL_DIR = "/storage/homefs/tj23t050/BERT/dnabert2_multiclass_model"
#OUTPUT_CSV = "/storage/homefs/tj23t050/BERT/190525/result/bert_snRNA_predictions_output.csv"

# dnabert2_human_tf_all_model
#MODEL_DIR = "/storage/homefs/tj23t050/BERT/dnabert2_human_tf_all_model"
#OUTPUT_CSV = "/storage/homefs/tj23t050/BERT/190525/result/human_tf_all_model_bert_snRNA_predictions_output.csv"

# dnabert2_prom_300_all_model
#MODEL_DIR = "/storage/homefs/tj23t050/BERT/dnabert2_prom_300_all_model"
#OUTPUT_CSV = "/storage/homefs/tj23t050/BERT/190525/result/prom_300_all_model_bert_snRNA_predictions_output.csv"

# dnabert2_prom_300_tata_model
#MODEL_DIR = "/storage/homefs/tj23t050/BERT/dnabert2_prom_300_tata_model"
#OUTPUT_CSV = "/storage/homefs/tj23t050/BERT/190525/result/prom_300_tata_model_bert_snRNA_predictions_output.csv"

# dnabert2_prom_core_all_model
#MODEL_DIR = "/storage/homefs/tj23t050/BERT/dnabert2_prom_core_all_model"
#OUTPUT_CSV = "/storage/homefs/tj23t050/BERT/190525/result/prom_core_all_model_bert_snRNA_predictions_output.csv"

# dnabert2_prom_core_tata_model
MODEL_DIR = "/storage/homefs/tj23t050/BERT/dnabert2_prom_core_tata_model"
OUTPUT_CSV = "/storage/homefs/tj23t050/BERT/190525/result/prom_core_tata_model_bert_snRNA_predictions_output.csv"

INPUT_CSV = "/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results/rfam_ensembl/-200to50_sequences.csv"

sys.path.insert(0, MODEL_DIR)
os.environ["DISABLE_TRITON"] = "1"

#################################################################################
# 
# # Label mapping (must match training)
label_list = ["no", "yes"] 
id_to_label = {0: "no", 1: "yes"}

#################################################################################
# 
# # Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, trust_remote_code=True)
model.eval()

#################################################################################
# 
# # Load input sequences
df = pd.read_csv(INPUT_CSV)  # Must have 'sequence' column
dataset = Dataset.from_pandas(df)

# Tokenize
def kmer_tokenize(sequence, k=6):
    return " ".join([sequence[i:i+k] for i in range(len(sequence)-k+1)])

def tokenize(example):
    return tokenizer(
        [kmer_tokenize(seq) for seq in example["sequence"]],
        padding="max_length",
        truncation=True,
        max_length=512
    )

print(tokenizer.vocab_size)
print(list(tokenizer.vocab.keys())[:10])

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

#################################################################################
# 
# # Predict
predictions = []
all_probs = []

with torch.no_grad():
    for batch in torch.utils.data.DataLoader(dataset, batch_size=8):
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        predictions.extend(preds)
        all_probs.extend(probs)

# Add predictions to DataFrame
df["prediction"] = predictions
df["label"] = df["prediction"].map(id_to_label)

# Add probabilities to DataFrame
probs_df = pd.DataFrame(all_probs, columns=[f"prob_{label}" for label in label_list])
df = pd.concat([df, probs_df], axis=1)

#################################################################################

SAVE_DIR = MODEL_DIR

with open(os.path.join(SAVE_DIR, "training_info.txt"), "w") as f:
    f.write("Training Task: Binary promoter classification (TATA)\n")
    f.write("Dataset: prom_core_tata\n")
    f.write("Labels: no, yes\n")
    f.write(f"Model base: {MODEL_DIR}\n")
    f.write(f"Model saved to: {SAVE_DIR}\n")
    f.write("Epochs: 6\n")
    f.write("Batch size: 4 (x2 accumulation)\n")
    f.write("Learning rate: 2e-5\n")
    f.write("Weight decay: 0.01\n")
    f.write("Warmup ratio: 0.1\n")
    f.write(f"Inference on {len(df)} sequences\n")
    f.write(f"Trained at: {datetime.now().isoformat()}\n")

    label_counts = df["label"].value_counts().to_dict()
    f.write(f"Label counts: {label_counts}\n")
    
print(df["label"].value_counts())
print(df[["sequence", "label", "prob_no", "prob_yes"]].head())

df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")
