# Credit Card Fraud Detection — Competitive ML 💳

Kaggle Playground Series S6E11 project for detecting fraudulent credit card transactions using **CatBoost** on a highly imbalanced tabular dataset.

## Result

| Metric | Score |
|---|---:|
| Local ROC-AUC | **0.944286** |
| Public Kaggle ROC-AUC | **0.94662** |
| Baseline PR-AUC | **0.509792** |

Dataset characteristics:

- **699,960** training transactions
- **1.67%** fraud rate
- mixed numerical, categorical and boolean features
- no missing values in the competition data

## Project overview

The project follows a complete tabular ML workflow:

1. data inspection and target analysis;
2. class imbalance assessment;
3. exploratory analysis of boolean risk signals;
4. feature-interaction analysis;
5. numerical feature analysis;
6. CatBoost baseline;
7. feature importance;
8. controlled ablation of the `id` column;
9. training on the full training set;
10. Kaggle submission.

## Class imbalance

Only **1.67%** of the training transactions are fraudulent.

Because of this, accuracy alone would be misleading. The main competition metric is **ROC-AUC**, while **PR-AUC** is also tracked as an additional diagnostic for the rare positive class.

## Exploratory findings

Several boolean signals showed strong associations with fraud.

Examples:

- `ip_country_mismatch=True` → ~16.5% fraud rate
- `is_foreign_transaction=True` → ~12.6%
- `billing_shipping_mismatch=True` → ~13.0%
- `is_ai_generated_scam_attempt=True` → ~15.2%

Interactions were even more informative:

- VPN + IP mismatch → ~43.7% fraud
- foreign transaction + IP mismatch → ~25.9%
- billing/shipping mismatch + IP mismatch → ~61.4%

The last interaction is especially strong but rare, so it should be interpreted together with its sample size.

## Numerical patterns

Important patterns observed during EDA included:

- higher `merchant_risk_score` among fraudulent transactions;
- higher `velocity_score`;
- strong increase in fraud rate with `cvv_retry_count`;
- higher `txn_count_last_24h`;
- lower `hours_since_last_txn`.

For example:

| CVV retries | Fraud rate |
|---:|---:|
| 0 | 0.93% |
| 1 | 8.75% |
| 2 | 41.73% |

## Model

The main model is a **CatBoostClassifier**, chosen because it handles mixed tabular data and categorical variables effectively while naturally capturing nonlinear relationships and feature interactions.

Main configuration:

- iterations: up to `1200`
- learning rate: `0.05`
- depth: `7`
- evaluation metric: `AUC`
- early stopping: `100`
- stratified 80/20 validation split
- random seed: `42`

### Baseline

The initial CatBoost model achieved:

- ROC-AUC: **0.944222**
- PR-AUC: **0.509792**
- best iteration: **930**

## Feature importance

Among the most relevant features identified by CatBoost were:

- `merchant_risk_score`
- `velocity_score`
- `customer_age`
- `cvv_retry_count`
- `auth_method`
- `is_new_merchant`

These results were consistent with several patterns found during exploratory analysis.

## ID ablation

Because `id` is an identifier and should not carry meaningful business information, a controlled ablation experiment was performed.

| Experiment | ROC-AUC | PR-AUC |
|---|---:|---:|
| CatBoost with `id` | 0.944222 | **0.509792** |
| CatBoost without `id` | **0.944286** | 0.508453 |

The ROC-AUC difference was only **+0.000064**, indicating that `id` did not provide meaningful predictive value.

The final submission therefore used the cleaner feature set **without `id`**.

## Kaggle submission

The final model was retrained on the full training set and used to generate probabilities for the 299,984-row test set.

**Public leaderboard ROC-AUC: 0.94662**

## Notebook

The complete documented analysis is available at [`notebooks/credit_card_fraud_competitive_ml.ipynb`](./notebooks/credit_card_fraud_competitive_ml.ipynb).

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── credit_card_fraud_competitive_ml.ipynb
└── src/
    ├── train.py
    └── predict.py
```

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python src/train.py --train path/to/train.csv
```

Generate predictions:

```bash
python src/predict.py --train path/to/train.csv --test path/to/test.csv --sample-submission path/to/sample_submission.csv
```

## Next steps

Potential improvements include:

- Stratified K-Fold and out-of-fold predictions;
- systematic feature engineering validated by cross-validation;
- comparison with LightGBM and XGBoost;
- hyperparameter optimization;
- model ensembling.

## Technologies

Python · Pandas · NumPy · Scikit-learn · CatBoost · Matplotlib · Kaggle

## Author

**Caio Vinicius Paes da Silva**

- GitHub: [caioooooo3](https://github.com/caioooooo3)
- Hugging Face: [caio16dev](https://huggingface.co/caio16dev)
- LinkedIn: [caio-viniciuss](https://www.linkedin.com/in/caio-viniciuss/)
