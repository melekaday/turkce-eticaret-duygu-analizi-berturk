"""Stage F: is the published two epoch recipe undertrained for the minority class?
Same hyperparameters, but training runs for four epochs with checkpoint selection driven by
the validation negative class F1. Mixed precision is used to keep the run tractable locally."""
import os, re, numpy as np, pandas as pd, torch
os.environ["HF_HUB_OFFLINE"] = "0"   # the base model weights are not in the local cache
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

OUT = os.path.dirname(os.path.abspath(__file__))
MODEL = "dbmdz/bert-base-turkish-uncased"

df = pd.DataFrame(load_dataset("fthbrmnby/turkish_product_reviews")["train"])
df["sentence"] = df["sentence"].apply(
    lambda t: re.sub(r"\s+", " ", re.sub(r"http\S+|www\S+", "", str(t).lower())).strip())
df = df[df["sentence"].apply(lambda x: len(x.split()) >= 2)].reset_index(drop=True)
X, y = df["sentence"], df["sentiment"].values
X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv)

tok = AutoTokenizer.from_pretrained(MODEL)


class DS(Dataset):
    def __init__(self, texts, labels):
        self.t, self.y = list(texts), list(labels)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        e = tok(self.t[i], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0),
                "attention_mask": e["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.y[i], dtype=torch.long)}


def compute_metrics(p):
    pred = np.argmax(p.predictions, axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, pred),
            "macro_f1": f1_score(p.label_ids, pred, average="macro"),
            "neg_f1": f1_score(p.label_ids, pred, pos_label=0)}


args = TrainingArguments(
    output_dir=os.path.join(OUT, "bert4_ckpt"),
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="neg_f1",
    greater_is_better=True,
    logging_steps=200,
    bf16=True,
    seed=42,
    report_to=[],
)

trainer = Trainer(model=AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2),
                  args=args, train_dataset=DS(X_tr, y_tr), eval_dataset=DS(X_va, y_va),
                  compute_metrics=compute_metrics)
trainer.train()

print("\n=== validation history ===", flush=True)
for h in trainer.state.log_history:
    if "eval_neg_f1" in h:
        print(f"  epoch {h['epoch']:.0f}  loss {h['eval_loss']:.4f}  acc {h['eval_accuracy']:.4f} "
              f"macroF1 {h['eval_macro_f1']:.4f}  negF1 {h['eval_neg_f1']:.4f}", flush=True)
print("best checkpoint:", trainer.state.best_model_checkpoint, "best val negF1:", trainer.state.best_metric)

for split, texts, labels in [("val", X_va, y_va), ("test", X_te, y_te)]:
    logits = trainer.predict(DS(texts, labels)).predictions
    proba = torch.softmax(torch.tensor(logits).float(), dim=-1).numpy()
    np.save(os.path.join(OUT, f"proba_bert4_{split}.npy"), proba)
    print(f"saved {split} probabilities", flush=True)
print("done")
