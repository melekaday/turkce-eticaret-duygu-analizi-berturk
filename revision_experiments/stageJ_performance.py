"""Stage J: performance of the resulting models beyond a single operating point.
(1) threshold free ranking quality on the test set, average precision for the negative class
    and ROC AUC, together with negative F1 by review length for the three final models,
(2) BERTurk operating points chosen on validation for a target negative recall,
(3) inference cost of the three final models on the 46,820 test reviews.
Timing needs the fine tuned weights (release assets, unzipped into models/)."""
import os, time, numpy as np, pandas as pd, joblib, scipy.sparse as sp, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OUT = os.path.dirname(os.path.abspath(__file__))
CLS = os.path.join(OUT, "models", "classical")
y_val = np.load(os.path.join(OUT, "y_val.npy"))
y_test = np.load(os.path.join(OUT, "y_test.npy"))
X_test = pd.read_parquet(os.path.join(OUT, "X_test.parquet"))["sentence"].tolist()

P = {
    "classical": (np.load(os.path.join(OUT, "proba_lr_val.npy")), np.load(os.path.join(OUT, "proba_lr_test.npy")), 0.35),
    "xlmr": (np.load(os.path.join(OUT, "proba_xlmr_val.npy"))[:, 1], np.load(os.path.join(OUT, "proba_xlmr_test.npy"))[:, 1], 0.50),
    "berturk": (np.load(os.path.join(OUT, "proba_berturk_val.npy"))[:, 1], np.load(os.path.join(OUT, "proba_berturk_test.npy"))[:, 1], 0.50),
}

# ---------- (1) threshold free ranking quality, the negative class is the class of interest ----------
neg = (y_test == 0).astype(int)
print("=== (1) threshold free ranking quality on the test set ===")
print(f"prevalence of the negative class (AP of a random ranking): {neg.mean():.4f}")
for k, (pv, pt, tau) in P.items():
    s = 1 - pt                                    # score for the negative class
    print(f"{k:10s} AP(neg) {average_precision_score(neg, s):.4f}   ROC AUC {roc_auc_score(neg, s):.4f}")

tok_b = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
lens = np.array([len(tok_b(t)["input_ids"]) for t in X_test])
print("\nnegative F1 by review length in BERTurk tokens, each model at its operating point")
print(f"{'bin':>10} {'n_all':>6} {'n_neg':>6} " + " ".join(f"{k:>10}" for k in P))
for lo, hi in [(0, 16), (16, 32), (32, 64), (64, 128), (128, 10**6)]:
    m = (lens > lo) & (lens <= hi)
    f = [f1_score(y_test[m], (P[k][1][m] >= P[k][2]).astype(int), pos_label=0) for k in P]
    print(f"{lo+1:>4}-{hi if hi < 10**6 else 'inf':>5} {m.sum():6d} {(y_test[m]==0).sum():6d} " + " ".join(f"{v:10.3f}" for v in f))

# ---------- (2) BERTurk operating points for a target negative recall, chosen on validation ----------
print("\n=== (2) BERTurk operating points, threshold chosen on validation for a target recall ===")
pv, pt, _ = P["berturk"]
grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
val_rec = np.array([recall_score(y_val, (pv >= t).astype(int), pos_label=0) for t in grid])
for target in (0.70, 0.75, 0.80, 0.85):
    tau = grid[np.argmax(val_rec >= target)]      # smallest threshold that reaches the target on validation
    pred = (pt >= tau).astype(int)
    fp = int(((pred == 0) & (y_test == 1)).sum())
    print(f"target recall {target:.2f}: tau {tau:.2f} | test negR {recall_score(y_test, pred, pos_label=0):.3f} "
          f"negP {precision_score(y_test, pred, pos_label=0):.3f} negF1 {f1_score(y_test, pred, pos_label=0):.3f} "
          f"missed negatives {int(((pred == 1) & (y_test == 0)).sum())} false alarms {fp} "
          f"({fp / (y_test == 1).sum() * 100:.2f}% of positives)")

# ---------- (3) inference cost ----------
print("\n=== (3) inference cost on the 46,820 test reviews ===")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu")

wv = joblib.load(os.path.join(CLS, "final_word_vectorizer.pkl"))
cv = joblib.load(os.path.join(CLS, "final_char_vectorizer.pkl"))
lr = joblib.load(os.path.join(CLS, "final_smote_lr_model.pkl"))
t0 = time.perf_counter()
_ = lr.predict_proba(sp.hstack([wv.transform(X_test), cv.transform(X_test)]).tocsr())
dt = time.perf_counter() - t0
print(f"classical, CPU, vectorization included: {dt:.1f} s  ({len(X_test) / dt:,.0f} reviews per second)")

MODELS = {"berturk": (os.environ.get("BERTURK_DIR", os.path.join(OUT, "models", "berturk_finetuned")), "dbmdz/bert-base-turkish-uncased"),
          "xlmr": (os.environ.get("XLMR_DIR", os.path.join(OUT, "models", "xlmr_finetuned")), "xlm-roberta-base")}


@torch.no_grad()
def timed_inference(ckpt, tok_name, texts, bs=128):
    tok = AutoTokenizer.from_pretrained(tok_name)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(DEV).eval()
    n_params = sum(p.numel() for p in model.parameters())
    warm = tok(texts[:bs], max_length=128, padding="max_length", truncation=True, return_tensors="pt").to(DEV)
    model(**warm)                                  # warm up
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], max_length=128, padding="max_length", truncation=True, return_tensors="pt").to(DEV)
        model(**b).logits.float().softmax(-1).cpu()
    if DEV == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return dt, n_params


for k, (ckpt, tn) in MODELS.items():
    dt, n = timed_inference(ckpt, tn, X_test)
    print(f"{k}, {DEV}, 32 bit, batch 128, tokenization included: {dt:.1f} s  "
          f"({len(X_test) / dt:,.0f} reviews per second), {n / 1e6:.1f} M parameters")
