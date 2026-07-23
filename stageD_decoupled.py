"""Stage D: two stage imbalance aware fine tuning.
Stage 1 representation is the existing BERTurk checkpoint trained on the full imbalanced data.
Stage 2 re-estimates the classifier on a class balanced subset of the training split,
then the decision threshold is calibrated on validation. Test is touched once per variant."""
import os, re, numpy as np, pandas as pd, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OUT = os.path.dirname(os.path.abspath(__file__))
CKPT = r"c:\Users\Sefa6\Desktop\TJMCS\results\bert_checkpoints\checkpoint-8780"
TOK = "dbmdz/bert-base-turkish-uncased"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- rebuild the exact splits ----
df = pd.DataFrame(load_dataset("fthbrmnby/turkish_product_reviews")["train"])
df["sentence"] = df["sentence"].apply(
    lambda t: re.sub(r"\s+", " ", re.sub(r"http\S+|www\S+", "", str(t).lower())).strip())
df = df[df["sentence"].apply(lambda x: len(x.split()) >= 2)].reset_index(drop=True)
X, y = df["sentence"], df["sentiment"].values
X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv)
X_tr, y_tr = X_tr.reset_index(drop=True), np.asarray(y_tr)

# ---- class balanced subset of the training split ----
rng = np.random.RandomState(42)
neg_idx = np.where(y_tr == 0)[0]
pos_idx = rng.choice(np.where(y_tr == 1)[0], size=len(neg_idx), replace=False)
bal_idx = np.concatenate([neg_idx, pos_idx])
rng.shuffle(bal_idx)
X_bal, y_bal = X_tr.iloc[bal_idx].tolist(), y_tr[bal_idx]
print(f"balanced stage 2 subset: {len(y_bal)} reviews, {int((y_bal==0).sum())} negative, {int((y_bal==1).sum())} positive")


@torch.no_grad()
def pooled(texts, bs=128):
    """penultimate representation the classification head sees"""
    out = np.empty((len(texts), 768), dtype=np.float32)
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], max_length=128, padding="max_length",
                truncation=True, return_tensors="pt").to(DEV)
        h = model.bert(**b).pooler_output
        out[i:i + bs] = h.cpu().numpy()
        if (i // bs) % 100 == 0:
            print(f"   {i}/{len(texts)}", flush=True)
    return out


tok = AutoTokenizer.from_pretrained(TOK)
model = AutoModelForSequenceClassification.from_pretrained(CKPT).to(DEV).eval()

feats = {}
for name, texts in [("bal", X_bal), ("val", X_va.tolist()), ("test", X_te.tolist())]:
    f = os.path.join(OUT, f"feat_{name}.npy")
    if os.path.exists(f):
        feats[name] = np.load(f)
    else:
        print(f" extracting {name} features ({len(texts)})", flush=True)
        feats[name] = pooled(texts)
        np.save(f, feats[name])

TAUS = np.arange(0.05, 0.96, 0.01)


def run(tag, clf):
    clf.fit(feats["bal"], y_bal)
    pv = clf.predict_proba(feats["val"])[:, 1]
    pt = clf.predict_proba(feats["test"])[:, 1]
    vf, tau = max((f1_score(y_va, (pv >= t).astype(int), pos_label=0), t) for t in TAUS)
    pred = (pt >= tau).astype(int)
    cm = confusion_matrix(y_te, pred, labels=[0, 1])
    print(f"{tag:34s} tau={tau:.2f} valNegF1={vf:.4f} | test acc {accuracy_score(y_te,pred)*100:.2f} "
          f"macroF1 {f1_score(y_te,pred,average='macro'):.4f} negP {precision_score(y_te,pred,pos_label=0):.4f} "
          f"negR {recall_score(y_te,pred,pos_label=0):.4f} negF1 {f1_score(y_te,pred,pos_label=0):.4f} "
          f"[TN {cm[0,0]} FP {cm[0,1]} FN {cm[1,0]} TP {cm[1,1]}]")
    return pred


print("\n=== stage 2 classifier re-estimation on the balanced subset ===")
print("reference  BERTurk alone, tuned threshold: negF1 0.7226 (acc 96.69, macroF1 0.8525)")
for C in [0.01, 0.1, 1.0, 10.0]:
    p = run(f"balanced logistic head C={C}", LogisticRegression(C=C, max_iter=2000, n_jobs=-1))
    np.save(os.path.join(OUT, f"pred_crt_C{C}.npy"), p)
