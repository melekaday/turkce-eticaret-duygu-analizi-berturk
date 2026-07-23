"""Stage C: test whether a probability level fusion of the contextual branch and the
character aware classical branch beats the best single model on the minority class.
Weights and threshold are tuned on validation only, the test set is touched once per variant."""
import os, numpy as np, pandas as pd, joblib, scipy.sparse as sp, itertools
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

RES = r"c:\Users\Sefa6\Desktop\TJMCS\results"
OUT = os.path.dirname(os.path.abspath(__file__))

y_val = np.load(os.path.join(OUT, "y_val.npy"))
y_test = np.load(os.path.join(OUT, "y_test.npy"))
X_val = pd.read_parquet(os.path.join(OUT, "X_val.parquet"))["sentence"]
X_test = pd.read_parquet(os.path.join(OUT, "X_test.parquet"))["sentence"]

# --- classical branch probabilities, P(positive) ---
f_lr_val, f_lr_test = os.path.join(OUT, "proba_lr_val.npy"), os.path.join(OUT, "proba_lr_test.npy")
if os.path.exists(f_lr_val):
    lr_val, lr_test = np.load(f_lr_val), np.load(f_lr_test)
else:
    wv = joblib.load(os.path.join(RES, "final_word_vectorizer.pkl"))
    cv = joblib.load(os.path.join(RES, "final_char_vectorizer.pkl"))
    lr = joblib.load(os.path.join(RES, "final_smote_lr_model.pkl"))
    lr_val = lr.predict_proba(sp.hstack([wv.transform(X_val), cv.transform(X_val)]))[:, 1]
    lr_test = lr.predict_proba(sp.hstack([wv.transform(X_test), cv.transform(X_test)]))[:, 1]
    np.save(f_lr_val, lr_val); np.save(f_lr_test, lr_test)

P = {
    "lr":      (lr_val, lr_test),
    "berturk": (np.load(os.path.join(OUT, "proba_berturk_val.npy"))[:, 1],
                np.load(os.path.join(OUT, "proba_berturk_test.npy"))[:, 1]),
    "xlmr":    (np.load(os.path.join(OUT, "proba_xlmr_val.npy"))[:, 1],
                np.load(os.path.join(OUT, "proba_xlmr_test.npy"))[:, 1]),
}
TAUS = np.arange(0.05, 0.96, 0.01)


def tune(p_val):
    """pick the threshold maximizing validation negative F1"""
    scores = [(f1_score(y_val, (p_val >= t).astype(int), pos_label=0), t) for t in TAUS]
    return max(scores)  # (val negF1, tau)


def evaluate(p_test, tau):
    pred = (p_test >= tau).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    return dict(acc=accuracy_score(y_test, pred) * 100,
                macro=f1_score(y_test, pred, average="macro"),
                negP=precision_score(y_test, pred, pos_label=0),
                negR=recall_score(y_test, pred, pos_label=0),
                negF1=f1_score(y_test, pred, pos_label=0),
                cm=cm, pred=pred)


def show(name, val_f1, tau, r):
    print(f"{name:38s} tau={tau:.2f} valNegF1={val_f1:.4f} | test acc {r['acc']:.2f} macroF1 {r['macro']:.4f} "
          f"negP {r['negP']:.4f} negR {r['negR']:.4f} negF1 {r['negF1']:.4f}  "
          f"[TN {r['cm'][0,0]} FP {r['cm'][0,1]} FN {r['cm'][1,0]} TP {r['cm'][1,1]}]")


print("=== single models, threshold tuned on validation ===")
singles = {}
for k, (pv, pt) in P.items():
    vf, tau = tune(pv)
    singles[k] = evaluate(pt, tau)
    show(k, vf, tau, singles[k])

print("\n=== two branch fusion, weight and threshold tuned on validation ===")
best_overall = None
for a, b in itertools.combinations(P, 2):
    best = None
    for w in np.arange(0.0, 1.01, 0.05):
        pv = w * P[a][0] + (1 - w) * P[b][0]
        vf, tau = tune(pv)
        if best is None or vf > best[0]:
            best = (vf, w, tau)
    vf, w, tau = best
    pt = w * P[a][1] + (1 - w) * P[b][1]
    r = evaluate(pt, tau)
    show(f"{w:.2f}*{a} + {1-w:.2f}*{b}", vf, tau, r)
    if best_overall is None or vf > best_overall[0]:
        best_overall = (vf, f"{w:.2f}*{a}+{1-w:.2f}*{b}", tau, r)

print("\n=== three branch fusion ===")
best3 = None
for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        w3 = 1 - w1 - w2
        pv = w1 * P["berturk"][0] + w2 * P["xlmr"][0] + w3 * P["lr"][0]
        vf, tau = tune(pv)
        if best3 is None or vf > best3[0]:
            best3 = (vf, (w1, w2, w3), tau)
vf, (w1, w2, w3), tau = best3
pt = w1 * P["berturk"][1] + w2 * P["xlmr"][1] + w3 * P["lr"][1]
r3 = evaluate(pt, tau)
show(f"{w1:.2f}*bert + {w2:.2f}*xlmr + {w3:.2f}*lr", vf, tau, r3)
if vf > best_overall[0]:
    best_overall = (vf, f"{w1:.2f}*bert+{w2:.2f}*xlmr+{w3:.2f}*lr", tau, r3)

print(f"\n=== best by validation: {best_overall[1]} (tau {best_overall[2]:.2f}) ===")
base = singles["berturk"]
print(f"BERTurk alone      negF1 {base['negF1']:.4f}  macroF1 {base['macro']:.4f}  acc {base['acc']:.2f}")
print(f"Proposed fusion    negF1 {best_overall[3]['negF1']:.4f}  macroF1 {best_overall[3]['macro']:.4f}  acc {best_overall[3]['acc']:.2f}")

# McNemar, proposed fusion against BERTurk alone
a_ok = (base["pred"] == y_test)
b_ok = (best_overall[3]["pred"] == y_test)
n01 = int((~a_ok & b_ok).sum()); n10 = int((a_ok & ~b_ok).sum())
res = mcnemar([[int((a_ok & b_ok).sum()), n10], [n01, int((~a_ok & ~b_ok).sum())]], exact=False, correction=True)
print(f"McNemar fusion vs BERTurk: only fusion correct {n01}, only BERTurk correct {n10}, "
      f"chi2 {res.statistic:.2f}, p {res.pvalue:.3e}")

np.save(os.path.join(OUT, "pred_fusion_test.npy"), best_overall[3]["pred"])
