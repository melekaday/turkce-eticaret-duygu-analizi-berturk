"""Stage E: is the 128 token limit a real bottleneck, and is truncation associated with
the misclassified negatives? Diagnostic only, no training."""
import os, numpy as np, pandas as pd
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer

OUT = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")

X_test = pd.read_parquet(os.path.join(OUT, "X_test.parquet"))["sentence"].tolist()
y_test = np.load(os.path.join(OUT, "y_test.npy"))
pred = np.load(os.path.join(OUT, "proba_berturk_test.npy")).argmax(1)

lens = np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in X_test])
print(f"token length  mean {lens.mean():.1f}  median {np.median(lens):.0f}  p90 {np.percentile(lens,90):.0f} "
      f"p95 {np.percentile(lens,95):.0f}  p99 {np.percentile(lens,99):.0f}  max {lens.max()}")
for L in (128, 192, 256, 384, 512):
    print(f"  reviews longer than {L:3d} tokens: {(lens > L).sum():6d}  ({(lens > L).mean()*100:.2f}%)")

neg = y_test == 0
miss = neg & (pred == 1)          # negative reviews predicted positive
hit = neg & (pred == 0)
print(f"\nnegative reviews: {neg.sum()}  correctly caught {hit.sum()}  missed {miss.sum()}")
print(f"  truncated (>128) among missed negatives : {(lens[miss] > 128).sum():4d}  ({(lens[miss]>128).mean()*100:.2f}%)")
print(f"  truncated (>128) among caught negatives : {(lens[hit] > 128).sum():4d}  ({(lens[hit]>128).mean()*100:.2f}%)")
print(f"  mean length missed {lens[miss].mean():.1f} vs caught {lens[hit].mean():.1f}")

# accuracy on the minority class split by length bucket
print("\nnegative class recall by review length")
for lo, hi in [(0, 16), (16, 32), (32, 64), (64, 128), (128, 10**6)]:
    m = neg & (lens > lo) & (lens <= hi)
    if m.sum():
        print(f"  {lo:3d}-{hi if hi<10**6 else 'inf':>4} tokens  n={m.sum():5d}  recall {(pred[m]==0).mean():.3f}")
