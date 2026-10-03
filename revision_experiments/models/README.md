# Models

* `classical/` holds the fitted word TF-IDF vectorizer, the character TF-IDF vectorizer, and the SMOTE plus Logistic Regression model of the article (decision threshold 0.35 on P(positive)).
* `berturk_finetuned/` and `xlmr_finetuned/` are not stored in git. Download `berturk_finetuned.zip` and `xlmr_finetuned.zip` from the releases of this repository and unzip them here. Each folder is a complete Hugging Face model directory with the weights, the configuration, the tokenizer files, and the training log (`trainer_state.json`).
