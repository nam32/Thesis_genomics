"""

Purpose:
-------
Evaluate a fine-tuned DNABERT-2 *multiclass* classifier on a hold-out split and
save accuracy/F1 metrics to JSON.

Workflow
--------
1) Load a CSV with columns: ['sequence', 'label'] (string labels).
2) Map string labels → integers consistently and save the validation split.
3) Tokenize raw sequences (DNABERT-2 uses BPE; do NOT k-mer).
4) Run evaluation via Hugging Face `Trainer`.
5) Save metrics to `<MODEL_DIR>/eval_metrics.json`.

Notes
-----
- Ensure the model’s internal label mapping matches your `label_to_id`. If the
  model was trained with a different order, set `model.config.id2label/label2id`
  before evaluation.
"""

import sys
import pandas as pd
import numpy as np
import json
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

##################################################################################

MODEL_DIR = "/storage/homefs/tj23t050/BERT/dnabert2_multiclass_model_2"
DATA_CSV = "/storage/homefs/tj23t050/BERT/training_data/dnabert2_promoter_multiclass.csv"

# Ensure custom modules are resolvable
sys.path.append(MODEL_DIR)

##################################################################################
# 
# # Load data
df = pd.read_csv(DATA_CSV)
label_list = sorted(df["label"].unique())
label_to_id = {label: i for i, label in enumerate(label_list)}
df["label"] = df["label"].map(label_to_id)

dataset = Dataset.from_pandas(df[['sequence', 'label']]).train_test_split(test_size=0.2, seed=42)
val_df = pd.DataFrame(dataset["test"])
val_df.to_csv("/storage/homefs/tj23t050/BERT/training_data/val_20_dnabert2_promoter_multiclass.csv", index=False)


##################################################################################
# 
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
def tokenize(example):
    return tokenizer(example["sequence"], padding="max_length", truncation=True, max_length=512)
dataset = dataset.map(tokenize, batched=True)

##################################################################################

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, trust_remote_code=True)

##################################################################################

# Evaluation function
def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

##################################################################################

# Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

##################################################################################

# Evaluate and save
metrics = trainer.evaluate()
print("Final metrics:", metrics)

with open(f"{MODEL_DIR}/eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
