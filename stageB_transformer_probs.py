"""Stage B: recover validation and test probabilities from the saved BERTurk and
XLM-RoBERTa checkpoints, verify they reproduce the published confusion matrices,
then tune the decision threshold on validation exactly as done for the classical models."""
import os, numpy as np, pandas as pd, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

RES = r"c:\Users\Sefa6\Desktop\TJMCS\results"
OUT = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

X_val = pd.read_parquet(os.path.join(OUT, "X_val.parquet"))["sentence"].tolist()
X_test = pd.read_parquet(os.path.join(OUT, "X_test.parquet"))["sentence"].tolist()
y_val = np.load(os.path.join(OUT, "y_val.npy"))
y_test = np.load(os.path.join(OUT, "y_test.npy"))

MODELS = {
    "berturk": (os.path.join(RES, "bert_checkpoints", "checkpoint-8780"), "dbmdz/bert-base-turkish-uncased"),
    "xlmr":    (os.path.join(RES, "xlmr_checkpoints", "checkpoint-8780"), "xlm-roberta-base"),
}


@torch.no_grad()
def predict_proba(ckpt, tok_name, texts, bs=128):
    tok = AutoTokenizer.from_pretrained(tok_name)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(DEV).eval()
    # full 32 bit precision, matching the precision used during fine tuning
    out = np.empty((len(texts), 2), dtype=np.float32)
    for i in range(0, len(texts), bs):
        batch = tok(texts[i:i + bs], max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt").to(DEV)
        logits = model(**batch).logits.float()
        out[i:i + bs] = torch.softmax(logits, dim=-1).cpu().numpy()
        if (i // bs) % 50 == 0:
            print(f"  {i + len(batch['input_ids'])}/{len(texts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def report(tag, y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    print(f"  {tag}: TN {cm[0,0]} FP {cm[0,1]} FN {cm[1,0]} TP {cm[1,1]} | "
          f"acc {accuracy_score(y, pred)*100:.2f} macroF1 {f1_score(y, pred, average='macro'):.4f} "
          f"negP {precision_score(y, pred, pos_label=0):.4f} negR {recall_score(y, pred, pos_label=0):.4f} "
          f"negF1 {f1_score(y, pred, pos_label=0):.4f}")
    return cm


for name, (ckpt, tok_name) in MODELS.items():
    print(f"\n================ {name} ================", flush=True)
    for split, texts, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        f = os.path.join(OUT, f"proba_{name}_{split}.npy")
        if os.path.exists(f):
            proba = np.load(f)
        else:
            print(f" running inference on {split} ({len(texts)} reviews)", flush=True)
            proba = predict_proba(ckpt, tok_name, texts)
            np.save(f, proba)
        print(f" {split}:")
        report("argmax", y, proba.argmax(1))

    # sanity check against the previously saved XLM-R predictions
    if name == "xlmr":
        old = np.load(os.path.join(RES, "y_pred_xlmr.npy"))
        new = np.load(os.path.join(OUT, "proba_xlmr_test.npy")).argmax(1)
        print(f" match with stored y_pred_xlmr.npy: {int((old == new).sum())}/{len(old)}")

    # threshold tuning on validation, same convention as the classical models:
    # p = P(positive), predict positive when p >= tau
    pv = np.load(os.path.join(OUT, f"proba_{name}_val.npy"))[:, 1]
    pt = np.load(os.path.join(OUT, f"proba_{name}_test.npy"))[:, 1]
    rows = []
    for tau in np.arange(0.10, 0.90, 0.05):
        f1 = f1_score(y_val, (pv >= tau).astype(int), pos_label=0)
        rows.append((round(float(tau), 2), f1))
    best_tau, best_f1 = max(rows, key=lambda r: r[1])
    print(f" validation scan (negative F1): " + "  ".join(f"{t}:{f:.4f}" for t, f in rows))
    print(f" best tau on validation = {best_tau} (val negF1 {best_f1:.4f})")
    print(" test at default 0.5 vs tuned:")
    report("tau=0.50", y_test, (pt >= 0.50).astype(int))
    report(f"tau={best_tau}", y_test, (pt >= best_tau).astype(int))

print("\nPaper targets  BERTurk test: TN 1964 FP 992 FN 536 TP 43328")
print("Paper targets  XLM-R   test: TN 1686 FP 1270 FN 615 TP 43249")
