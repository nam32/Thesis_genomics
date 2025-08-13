"""

Purpose:
-------
Fine-tune a local DNABERT-2 model on promoter classification datasets
(variants: prom_300_all, prom_core_all, prom_300_tata, prom_core_tata),
then evaluate on a held-out test split and save artefacts (model, tokenizer,
classification report, confusion matrix).

What it does
------------
1) Loads train/dev/test CSVs with columns: ['sequence', 'label'].
2) Builds a consistent label mapping (string → int) across all splits.
3) Tokenizes raw DNA sequences with DNABERT-2’s BPE tokenizer
   (no k-mer slicing; DNABERT-2 is not k-mer based).
4) Loads a local DNABERT-2 backbone (MODEL_DIR) for sequence classification,
   sets num_labels, and trains with HF Trainer.
5) Tracks metrics each epoch (accuracy, weighted F1), saves the best model.
6) Evaluates on the test set; writes a detailed classification report and confusion matrix.

Inputs to edit
--------------
- MODEL_DIR : path to local DNABERT-2 weights (backbone).
- SAVE_DIR  : where the fine-tuned model and outputs will be saved.
- train/dev/test CSVs : pick one dataset family (uncomment the three paths).

Key settings
------------
- Tokenizer: zhihan1996/DNABERT-2-117M (trust_remote_code=True).
- Max length: 512, padding to max_length, truncation enabled.
- Training: epochs=30, per_device_batch=4, grad_accum=8 (effective batch ≈ 32).
- Selection: load_best_model_at_end=True, metric_for_best_model="f1".
- Env: os.environ["DISABLE_TRITON"]="1" to avoid Triton kernels on some clusters.

Outputs
-------
- SAVE_DIR/
  ├─ config.json / pytorch_model.bin / tokenizer files
  ├─ classification_report.txt         (per-class precision/recall/F1, macro/weighted)
  ├─ confusion_matrix.png              (test set)
  └─ trainer logs and checkpoints      (under output_dir)

Notes & tips
------------
- Ensure label strings are identical across splits before mapping.
- If you change the dataset family, switch all three CSV paths together.
- For stability on GPUs with limited memory, tune per_device_* and grad_accum.
- To make labels human-readable in downstream tools, you may set:
    model.config.label2id = label_to_id
    model.config.id2label = {v:k for k,v in label_to_id.items()}
"""

import os
os.environ["DISABLE_TRITON"] = "1"

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig
from datasets import Dataset, DatasetDict
import sys

########################################################################################

# Paths
MODEL_DIR = "/storage/homefs/tj23t050/BERT/190525/local_dnabert2"
#SAVE_DIR = "/storage/homefs/tj23t050/BERT/dnabert2_multiclass_model"
#DATA_CSV = "/storage/homefs/tj23t050/BERT/training_data/dnabert2_promoter_multiclass.csv"

## DNABERT Datasets
#SAVE_DIR = "/storage/homefs/tj23t050/BERT/dnabert2_prom_300_all_model"
#SAVE_DIR = "/storage/homefs/tj23t050/BERT/dnabert2_prom_core_all_model"
SAVE_DIR = "/storage/homefs/tj23t050/BERT/dnabert2_prom_300_tata_model_2"
#SAVE_DIR = "/storage/homefs/tj23t050/BERT/dnabert2_prom_core_tata_model"

sys.path.append("/storage/homefs/tj23t050/BERT/190525/local_dnabert2")

########################################################################################

# Step 1: Load and encode data

# Load all three splits
#train_df = pd.read_csv("/storage/homefs/tj23t050/BERT/training_data/prom_300_all/prom_300_all_train.csv")
#dev_df   = pd.read_csv("/storage/homefs/tj23t050/BERT/training_data/prom_300_all/prom_300_all_dev.csv")
#test_df  = pd.read_csv("/storage/homefs/tj23t050/BERT/training_data/prom_300_all/prom_300_all_test.csv")

# Alternative dataset paths (commented)
#train_df = pd.read_csv("/storage/homefs/tj23t050/BERT/training_data/prom_core_all/prom_core_all_train.csv")
#dev_df   = pd.read_csv("/storage/homefs/tj23t050/BERT/training_data/prom_core_all/prom_core_all_dev.csv")
#test_df  = pd.read_csv("/storage/homefs/tj23t050/BERT/training_data/prom_core_all/prom_core_all_test.csv")

train_df = pd.read_csv("/storage/homefs/tj23t050/BERT/training_data/prom_300_tata/prom_300_tata_train.csv")
dev_df   = pd.read_csv("/storage/homefs/tj23t050/BERT/training_data/prom_300_tata/prom_300_tata_dev.csv")
test_df  = pd.read_csv("/storage/homefs/tj23t050/BERT/training_data/prom_300_tata/prom_300_tata_test.csv")

# train_df = pd.read_csv("/storage/homefs/tj23t050/BERT/training_data/prom_core_tata/prom_core_tata_train.csv")
# dev_df   = pd.read_csv("/storage/homefs/tj23t050/BERT/training_data/prom_core_tata/prom_core_tata_dev.csv")
# test_df  = pd.read_csv("/storage/homefs/tj23t050/BERT/training_data/prom_core_tata/prom_core_tata_test.csv")

print("Training Task: dnabert2_prom_300_tata_model classification")
print("Dataset: prom_300_tata_train.csv")
print(f"Model save path: {SAVE_DIR}")

# Map labels
label_list = sorted(train_df["label"].unique())
label_to_id = {label: i for i, label in enumerate(label_list)}
for df in [train_df, dev_df, test_df]:
    df["label"] = df["label"].map(label_to_id)

# Retain only required columns to reduce memory
train_df = train_df[["sequence", "label"]]
dev_df = dev_df[["sequence", "label"]]
test_df = test_df[["sequence", "label"]]

# Convert to Hugging Face Datasets
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(dev_df, preserve_index=False),
    "test": Dataset.from_pandas(test_df, preserve_index=False),
})

num_labels = len(label_list)

########################################################################################

# Step 2: Tokenizer
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

def tokenize(example):
    return tokenizer(example["sequence"], padding="max_length", truncation=True, max_length=512)

# Memory-efficient mapping: non-batched + drop raw text column
dataset = dataset.map(tokenize, batched=False, remove_columns=["sequence"])

########################################################################################

# Step 3: Load local DNABERT-2 model
from transformers import AutoModelForSequenceClassification

config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
config.num_labels = num_labels

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    config=config,
    local_files_only=True,
    trust_remote_code=True
)

########################################################################################

# Step 5: Training
def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    if isinstance(logits, tuple):  # handle (logits,) or other formats
        logits = logits[0]
    preds = np.argmax(logits, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=30,
    per_device_train_batch_size=4,         # Reduced for memory
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,         # Effective batch size = 8
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].with_format("torch"),
    eval_dataset=dataset["validation"].with_format("torch"),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

########################################################################################

# Step 6: Train + Save
trainer.train()
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Predict on test set
predictions_output = trainer.predict(dataset["test"].with_format("torch"))
logits = predictions_output.predictions
if isinstance(logits, tuple):
    logits = logits[0]
preds = np.argmax(logits, axis=1)
labels = predictions_output.label_ids

# Convert label ids to original string labels
id_to_label = {v: k for k, v in label_to_id.items()}
true_labels = [id_to_label[i] for i in labels]
pred_labels = [id_to_label[i] for i in preds]

# Classification report
report = classification_report(true_labels, pred_labels, target_names=label_list, digits=4)
print("Classification Report:\n")
print(report)

# Save report
with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=label_list)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.show()
