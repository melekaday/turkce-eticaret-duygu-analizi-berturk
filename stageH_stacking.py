"""Stage H: an error analysis informed meta classifier over the two branches.
The naive weighted average failed, so here the combination is learned and is given the
linguistic cues that the error taxonomy identified as the failure modes.
The validation set is split in half so that the meta classifier and its threshold never
see the same data, and the test set is touched once."""
import os, re, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

OUT = os.path.dirname(os.path.abspath(__file__))
y_val = np.load(os.path.join(OUT, "y_val.npy"))
y_test = np.load(os.path.join(OUT, "y_test.npy"))
X_val = pd.read_parquet(os.path.join(OUT, "X_val.parquet"))["sentence"].tolist()
X_test = pd.read_parquet(os.path.join(OUT, "X_test.parquet"))["sentence"].tolist()

NEG = ["değil", "yok", "hiç", "olmaz", "olmadı", "etmedi", "vermedi"]
POSW = ["iyi", "güzel", "kaliteli", "mükemmel", "harika", "süper"]
NEGW = ["kötü", "berbat", "rezalet", "pahalı", "bozuk", "yavaş"]


def features(texts, p_bert, p_xlmr, p_lr):
    n = len(texts)
    F = np.zeros((n, 11), dtype=np.float32)
    F[:, 0], F[:, 1], F[:, 2] = p_bert, p_xlmr, p_lr
    F[:, 3] = np.abs(p_bert - p_lr)
    F[:, 4] = np.abs(p_bert - p_xlmr)
    F[:, 5] = np.abs(p_bert - 0.5)                      # confidence of the contextual branch
    for i, t in enumerate(texts):
        w = t.split()
        F[i, 6] = len(w)
        F[i, 7] = any(k in t for k in NEG)               # negation, 27.7% of the residual errors
        F[i, 8] = any(len(x) > 15 for x in w)            # joined or misspelled tokens
        F[i, 9] = any(k in t for k in POSW) and any(k in t for k in NEGW)   # mixed sentiment
        F[i, 10] = t.count("!") + t.count("?")
    return F


P = {k: (np.load(os.path.join(OUT, f"proba_{k}_val.npy")), np.load(os.path.join(OUT, f"proba_{k}_test.npy")))
     for k in ("berturk", "xlmr")}
lr_v, lr_t = np.load(os.path.join(OUT, "proba_lr_val.npy")), np.load(os.path.join(OUT, "proba_lr_test.npy"))

Fv = features(X_val, P["berturk"][0][:, 1], P["xlmr"][0][:, 1], lr_v)
Ft = features(X_test, P["berturk"][1][:, 1], P["xlmr"][1][:, 1], lr_t)

# half of validation trains the meta classifier, the other half calibrates its threshold
iA, iB = train_test_split(np.arange(len(y_val)), test_size=0.5, random_state=42, stratify=y_val)
TAUS = np.arange(0.02, 0.99, 0.01)

# baseline: BERTurk with its threshold calibrated on the same held out half
pb_val, pb_test = P["berturk"][0][:, 1], P["berturk"][1][:, 1]
_, tau_b = max((f1_score(y_val[iB], (pb_val[iB] >= t).astype(int), pos_label=0), t) for t in TAUS)
pred_base = (pb_test >= tau_b).astype(int)


def report(tag, pred):
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    print(f"{tag:44s} acc {accuracy_score(y_test,pred)*100:.2f}  macroF1 {f1_score(y_test,pred,average='macro'):.4f}  "
          f"negP {precision_score(y_test,pred,pos_label=0):.4f}  negR {recall_score(y_test,pred,pos_label=0):.4f}  "
          f"negF1 {f1_score(y_test,pred,pos_label=0):.4f}  [TN {cm[0,0]} FP {cm[0,1]} FN {cm[1,0]} TP {cm[1,1]}]")


report(f"BERTurk baseline (tau {tau_b:.2f})", pred_base)

FEATSETS = {
    "probabilities only":        [0, 1, 2],
    "probabilities + agreement": [0, 1, 2, 3, 4, 5],
    "all, with linguistic cues": list(range(11)),
}
best = None
for fname, cols in FEATSETS.items():
    for mname, mk in [("logistic", lambda: LogisticRegression(max_iter=2000, class_weight="balanced")),
                      ("gradient boosting", lambda: HistGradientBoostingClassifier(random_state=42))]:
        clf = mk().fit(Fv[iA][:, cols], y_val[iA])
        pB = clf.predict_proba(Fv[iB][:, cols])[:, 1]
        vf, tau = max((f1_score(y_val[iB], (pB >= t).astype(int), pos_label=0), t) for t in TAUS)
        pred = (clf.predict_proba(Ft[:, cols])[:, 1] >= tau).astype(int)
        report(f"{mname}, {fname} (tau {tau:.2f})", pred)
        if best is None or vf > best[0]:
            best = (vf, f"{mname}, {fname}", pred)

print(f"\nbest by held out validation half: {best[1]}")
a_ok, b_ok = pred_base == y_test, best[2] == y_test
n01, n10 = int((~a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())
r = mcnemar([[int((a_ok & b_ok).sum()), n10], [n01, int((~a_ok & ~b_ok).sum())]], exact=False, correction=True)
print(f"McNemar vs BERTurk baseline: only meta correct {n01}, only baseline correct {n10}, "
      f"chi2 {r.statistic:.2f}, p {r.pvalue:.3e}")
