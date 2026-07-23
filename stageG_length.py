"""Stage G: do the two branches fail on different length regimes?
If the character aware classical branch is relatively stronger on short reviews,
a length gated combination is motivated rather than arbitrary."""
import os, numpy as np, pandas as pd
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

OUT = os.path.dirname(os.path.abspath(__file__))
y_val = np.load(os.path.join(OUT, "y_val.npy"))
y_test = np.load(os.path.join(OUT, "y_test.npy"))
X_val = pd.read_parquet(os.path.join(OUT, "X_val.parquet"))["sentence"]
X_test = pd.read_parquet(os.path.join(OUT, "X_test.parquet"))["sentence"]
wv = np.array([len(s.split()) for s in X_val])
wt = np.array([len(s.split()) for s in X_test])

bert = (np.load(os.path.join(OUT, "proba_berturk_val.npy"))[:, 1],
        np.load(os.path.join(OUT, "proba_berturk_test.npy"))[:, 1])
lr = (np.load(os.path.join(OUT, "proba_lr_val.npy")), np.load(os.path.join(OUT, "proba_lr_test.npy")))
TAUS = np.arange(0.01, 0.99, 0.01)


def tune(p, y, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    return max((f1_score(y[m], (p[m] >= t).astype(int), pos_label=0), t) for t in TAUS)


tb, taub = tune(bert[0], y_val)
tl, taul = tune(lr[0], y_val)
print(f"validation tuned thresholds: BERTurk {taub:.2f}, classical {taul:.2f}\n")

print("negative class F1 by review length, tuned per branch on the whole validation set")
print(f"{'bucket':>12}  {'n_test':>7}  {'n_neg':>6}  {'BERTurk':>8}  {'classical':>9}")
BUCKETS = [(0, 3), (3, 6), (6, 12), (12, 25), (25, 10**6)]
for lo, hi in BUCKETS:
    m = (wt > lo) & (wt <= hi)
    pb = (bert[1][m] >= taub).astype(int)
    pl = (lr[1][m] >= taul).astype(int)
    lab = f"{lo+1}-{hi if hi < 10**6 else 'inf'}"
    print(f"{lab:>12}  {m.sum():7d}  {(y_test[m]==0).sum():6d}  "
          f"{f1_score(y_test[m], pb, pos_label=0):8.4f}  {f1_score(y_test[m], pl, pos_label=0):9.4f}")

# length gated routing: choose the branch per bucket using the validation set only
print("\nlength gated routing, branch chosen per bucket on validation")
pred_test = np.empty(len(y_test), int)
for lo, hi in BUCKETS:
    mv, mt = (wv > lo) & (wv <= hi), (wt > lo) & (wt <= hi)
    fb, tb_ = tune(bert[0], y_val, mv)
    fl, tl_ = tune(lr[0], y_val, mv)
    use_bert = fb >= fl
    print(f"  bucket {lo+1}-{hi if hi < 10**6 else 'inf':>3}: val negF1 BERTurk {fb:.4f} (tau {tb_:.2f}) vs "
          f"classical {fl:.4f} (tau {tl_:.2f}) -> {'BERTurk' if use_bert else 'classical'}")
    src, tau = (bert[1], tb_) if use_bert else (lr[1], tl_)
    pred_test[mt] = (src[mt] >= tau).astype(int)

print(f"\nreference BERTurk, single tuned threshold : negF1 0.7226  macroF1 0.8525  acc 96.69")
print(f"length gated routing                      : negF1 {f1_score(y_test, pred_test, pos_label=0):.4f}  "
      f"macroF1 {f1_score(y_test, pred_test, average='macro'):.4f}  acc {accuracy_score(y_test, pred_test)*100:.2f}  "
      f"negP {precision_score(y_test, pred_test, pos_label=0):.4f} negR {recall_score(y_test, pred_test, pos_label=0):.4f}")

# per bucket thresholds for a single branch, without any routing
print("\nper bucket threshold calibration of BERTurk alone")
pred2 = np.empty(len(y_test), int)
for lo, hi in BUCKETS:
    mv, mt = (wv > lo) & (wv <= hi), (wt > lo) & (wt <= hi)
    _, t = tune(bert[0], y_val, mv)
    pred2[mt] = (bert[1][mt] >= t).astype(int)
print(f"BERTurk with per bucket thresholds        : negF1 {f1_score(y_test, pred2, pos_label=0):.4f}  "
      f"macroF1 {f1_score(y_test, pred2, average='macro'):.4f}  acc {accuracy_score(y_test, pred2)*100:.2f}")
