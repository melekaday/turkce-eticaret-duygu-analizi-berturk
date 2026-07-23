"""Stage A: rebuild the exact splits and verify the threshold/label convention
against the saved SMOTE+LR model. CPU only, fast."""
import os, re, numpy as np, joblib, scipy.sparse as sp
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

RES = r"c:\Users\Sefa6\Desktop\TJMCS\results"
OUT = os.path.dirname(os.path.abspath(__file__))

ds = load_dataset("fthbrmnby/turkish_product_reviews")
df = pd.DataFrame(ds["train"])

def preprocess(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    return re.sub(r"\s+", " ", t).strip()

df["sentence"] = df["sentence"].apply(preprocess)
df = df[df["sentence"].apply(lambda x: len(x.split()) >= 2)].reset_index(drop=True)
print("after preprocessing:", len(df), "pos:", int((df.sentiment == 1).sum()), "neg:", int((df.sentiment == 0).sum()))

X, y = df["sentence"], df["sentiment"].values
X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv)
print("train/val/test:", len(y_tr), len(y_va), len(y_te))
print("test neg:", int((y_te == 0).sum()), "test pos:", int((y_te == 1).sum()))

np.save(os.path.join(OUT, "y_val.npy"), y_va)
np.save(os.path.join(OUT, "y_test.npy"), y_te)
X_va.to_frame().to_parquet(os.path.join(OUT, "X_val.parquet"))
X_te.to_frame().to_parquet(os.path.join(OUT, "X_test.parquet"))

# --- verify threshold convention with the saved classical model ---
wv = joblib.load(os.path.join(RES, "final_word_vectorizer.pkl"))
cv = joblib.load(os.path.join(RES, "final_char_vectorizer.pkl"))
lr = joblib.load(os.path.join(RES, "final_smote_lr_model.pkl"))
print("LR classes_:", lr.classes_)

Xte = sp.hstack([wv.transform(X_te), cv.transform(X_te)])
p_col1 = lr.predict_proba(Xte)[:, 1]          # P(label 1) = P(positive)

for label, pred in [
    ("argmax (0.5)", (p_col1 >= 0.5).astype(int)),
    ("p_pos >= 0.35 -> positive", (p_col1 >= 0.35).astype(int)),
    ("p_neg >= 0.35 -> negative", (p_col1 < 0.65).astype(int) * 0 + np.where(1 - p_col1 >= 0.35, 0, 1)),
]:
    cm = confusion_matrix(y_te, pred, labels=[0, 1])
    print(f"\n--- {label} ---")
    print("           pred_neg  pred_pos")
    print(f"true_neg   {cm[0,0]:8d}  {cm[0,1]:8d}")
    print(f"true_pos   {cm[1,0]:8d}  {cm[1,1]:8d}")
    print(f"acc {accuracy_score(y_te, pred):.4f}  macroF1 {f1_score(y_te, pred, average='macro'):.4f} "
          f"negP {precision_score(y_te, pred, pos_label=0):.4f} negR {recall_score(y_te, pred, pos_label=0):.4f} "
          f"negF1 {f1_score(y_te, pred, pos_label=0):.4f}")

print("\nPaper Table 7 target: 1875 / 1081 / 1346 / 42518, acc 94.82, macroF1 0.79, negP 0.58 negR 0.63 negF1 0.61")
