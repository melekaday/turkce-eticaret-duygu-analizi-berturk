"""Stage I: data behind the figures added in the minor revision.
(1) validation threshold curves of the three final models,
(2) negative class recall by review length for the three final models,
(3) the six decision level differences against the run to run spread.
Everything is computed from the stored probabilities, no training or inference.
The pgfplots coordinates printed here are pasted unchanged into the manuscript."""
import os, numpy as np, pandas as pd
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sklearn.metrics import f1_score, recall_score
from transformers import AutoTokenizer

OUT = os.path.dirname(os.path.abspath(__file__))
y_val = np.load(os.path.join(OUT, "y_val.npy"))
y_test = np.load(os.path.join(OUT, "y_test.npy"))
X_test = pd.read_parquet(os.path.join(OUT, "X_test.parquet"))["sentence"].tolist()

# P(positive) for every model, predict positive when p >= tau
P = {
    "classical": (np.load(os.path.join(OUT, "proba_lr_val.npy")), np.load(os.path.join(OUT, "proba_lr_test.npy"))),
    "xlmr": (np.load(os.path.join(OUT, "proba_xlmr_val.npy"))[:, 1], np.load(os.path.join(OUT, "proba_xlmr_test.npy"))[:, 1]),
    "berturk": (np.load(os.path.join(OUT, "proba_berturk_val.npy"))[:, 1], np.load(os.path.join(OUT, "proba_berturk_test.npy"))[:, 1]),
}
# operating points used in the paper: tuned threshold for the classical model, default cutoff for the transformers
OP = {"classical": 0.35, "xlmr": 0.50, "berturk": 0.50}


def neg_f1(y, p, tau):
    return f1_score(y, (p >= tau).astype(int), pos_label=0)


# ---------- (1) validation threshold curves on the paper grid [0.10, 0.85] ----------
GRID = np.round(np.arange(0.10, 0.8501, 0.01), 2)
print("=== (1) validation negative F1 against the threshold ===")
for k, (pv, pt) in P.items():
    curve = [neg_f1(y_val, pv, t) for t in GRID]
    i = int(np.argmax(curve))
    print(f"{k:10s} best tau {GRID[i]:.2f}  val negF1 {curve[i]:.4f}  at 0.50 {curve[list(GRID).index(0.5)]:.4f}  "
          f"range {min(curve):.4f}..{max(curve):.4f}  | test negF1 at best tau {neg_f1(y_test, pt, GRID[i]):.4f}")
    print("  coords: " + " ".join(f"({t:.2f},{c:.4f})" for t, c in zip(GRID, curve) if round(t * 100) % 2 == 0 or t == GRID[i]))

# ---------- (2) negative recall by review length, BERTurk tokens with special tokens ----------
tok = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
lens = np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in X_test])
BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 10**6)]
neg = y_test == 0
print("\n=== (2) negative class recall by review length in BERTurk tokens ===")
print(f"{'bin':>10} {'n_neg':>6} " + " ".join(f"{k:>10}" for k in P))
for lo, hi in BINS:
    m = neg & (lens > lo) & (lens <= hi)
    rec = {k: float(((P[k][1][m] >= OP[k]).astype(int) == 0).mean()) for k in P}
    print(f"{lo+1:>4}-{hi if hi < 10**6 else 'inf':>5} {m.sum():6d} " + " ".join(f"{rec[k]:10.3f}" for k in P))
print("overall    " + str(int(neg.sum())).rjust(6) + " " + " ".join(
    f"{recall_score(y_test, (P[k][1] >= OP[k]).astype(int), pos_label=0):10.3f}" for k in P))

# ---------- (3) run to run spread on the test set ----------
print("\n=== (3) run to run spread ===")
b4v, b4t = np.load(os.path.join(OUT, "proba_bert4_val.npy"))[:, 1], np.load(os.path.join(OUT, "proba_bert4_test.npy"))[:, 1]
G2 = np.round(np.arange(0.05, 0.9501, 0.01), 2)
for name, pv, pt in [("run 1", P["berturk"][0], P["berturk"][1]), ("run 2", b4v, b4t)]:
    vf, tau = max((neg_f1(y_val, pv, t), t) for t in G2)
    print(f"{name}: own validation tau {tau:.2f}  test negF1 {neg_f1(y_test, pt, tau):.4f}  "
          f"default cutoff test negF1 {neg_f1(y_test, pt, 0.5):.4f}")
base = neg_f1(y_test, P["berturk"][1], 0.5)
print(f"uncorrected BERTurk test negF1 {base:.4f}")
