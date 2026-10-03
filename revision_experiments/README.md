# Revision experiments

This folder contains every experiment added to the article

> M. Aday, S. Aras, *A minority-first protocol for Turkish sentiment analysis with transformer language models*, Turkish Journal of Mathematics and Computer Science.

on top of the main pipeline of this repository. Each script reads the stored probabilities of the three final models, so all reported numbers can be reproduced without repeating the fine-tuning. The fine-tuned BERTurk and XLM-RoBERTa models are attached to the [releases](https://github.com/melekaday/turkce-eticaret-duygu-analizi-berturk/releases) of this repository.

## Contents

| Path | What it is |
|---|---|
| `stageA_check.py` ... `stageJ_performance.py` | The experiment scripts, listed below |
| `proba_berturk_{val,test}.npy` | Softmax output of the fine-tuned BERTurk, shape (46820, 2), column 1 is P(positive) |
| `proba_xlmr_{val,test}.npy` | The same for the fine-tuned XLM-RoBERTa |
| `proba_lr_{val,test}.npy` | P(positive) of the best classical model, SMOTE plus Logistic Regression on Word plus Char TF-IDF |
| `proba_bert4_{val,test}.npy` | Softmax output of the repeated BERTurk run (16-bit, four epochs), used for the run-to-run spread |
| `y_pred_xlmr.npy` | XLM-RoBERTa test predictions saved by the original notebook, used as a consistency check |
| `models/classical/` | Fitted word and character TF-IDF vectorizers and the SMOTE plus Logistic Regression model |
| `models/berturk_finetuned/`, `models/xlmr_finetuned/` | Not in git. Unzip the release assets here (see below) |

Labels follow the dataset: `0` is negative, `1` is positive. Every decision rule predicts positive when P(positive) is at least the threshold.

## Setup

```bash
pip install -r requirements.txt
```

The scripts were run with Python 3.12, PyTorch 2.6 (CUDA 12.4), and the package versions in `requirements.txt`, on an NVIDIA GeForce RTX 4060.

To use the fine-tuned transformers, download `berturk_finetuned.zip` and `xlmr_finetuned.zip` from the release and unzip them into `models/`, so that `models/berturk_finetuned/config.json` exists. The folders can also be placed elsewhere and passed with the environment variables `BERTURK_DIR` and `XLMR_DIR`.

## Order of execution

`stageA_check.py` must run first. It downloads the `fthbrmnby/turkish_product_reviews` dataset from the Hugging Face Hub, applies the preprocessing of the article, rebuilds the stratified 60/20/20 split with seed 42 (140,460 / 46,820 / 46,820 reviews), and writes `X_val.parquet`, `X_test.parquet`, `y_val.npy`, and `y_test.npy`. It also checks the stored classical model against the confusion matrix of the article. The first run needs internet access. Afterwards the scripts work offline.

## Scripts and what they reproduce

| Script | Needs GPU | Reproduces |
|---|---|---|
| `stageA_check.py` | no | Data split, confusion matrix of the best classical model (panel (a) of Figure 3) |
| `stageB_transformer_probs.py` | only if the probability files are missing | Confusion matrices of both transformers (Figure 3), threshold calibration of the transformers (Table 15) |
| `stageC_fusion.py` | no | Probability fusion of two and three branches (Table 16) |
| `stageD_decoupled.py` | yes | Balanced re-estimation of the classifier on a class-balanced subset (Table 16) |
| `stageE_diag.py` | no | Token length statistics, truncation analysis, recall by review length (Table 21, Section 5.2) |
| `stageF_train4.py` | yes, several hours | Repeated BERTurk fine-tuning with four epochs (Table 17). Writes `proba_bert4_*.npy` |
| `stageG_length.py` | no | Length-gated routing and per-segment thresholds (Table 16) |
| `stageH_stacking.py` | no | Error-informed meta-classifier and its McNemar test (Table 16) |
| `stageI_figures.py` | no | Data of Figures 4, 5, and 7 (threshold curves, run-to-run spread, length-stratified performance) |
| `stageJ_performance.py` | yes, for the timing part | Average precision and ROC AUC (Table 22), operating points of BERTurk (Table 23), inference cost (Section 5.3) |

Example:

```bash
python stageA_check.py
python stageB_transformer_probs.py
python stageI_figures.py
```

## Using the fine-tuned model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

path = "models/berturk_finetuned"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path).eval()

texts = ["ürün çok güzel, hızlı kargo", "hiç beğenmedim, para çöpe gitti"]
batch = tok([t.lower() for t in texts], max_length=128, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    p_pos = torch.softmax(model(**batch).logits, -1)[:, 1]
tau = 0.67   # threshold calibrated on the validation set, 0.50 is the default cutoff
print([("positive" if p >= tau else "negative", round(float(p), 3)) for p in p_pos])
```

Inference must run in 32-bit precision to reproduce the published numbers exactly. Half precision changes the prediction of individual reviews.

## Notes

* `stageF_train4.py` needs the base model `dbmdz/bert-base-turkish-uncased` and therefore internet access on its first run.
* The scripts write their intermediate files (`X_*.parquet`, `y_*.npy`, `feat_*.npy`, `pred_*.npy`) next to themselves. These are listed in `.gitignore`.
* The classical model was fitted with scikit-learn 1.6.1. Newer versions load it with a warning and give identical predictions.
