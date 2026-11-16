"""Train a CatBoost baseline for the Kaggle credit card fraud competition."""

import argparse
import json
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

TARGET = "is_fraud"
RANDOM_STATE = 42


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Use the original competition features and remove only the identifier."""
    return df.drop(columns=["id"], errors="ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--model-out", default="catboost_fraud.cbm")
    args = parser.parse_args()

    train = pd.read_csv(args.train)

    X = build_features(train.drop(columns=[TARGET]))
    y = train[TARGET].astype(int)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.05,
        depth=7,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
    )

    pred = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_valid, pred),
        "pr_auc": average_precision_score(y_valid, pred),
        "best_iteration": model.get_best_iteration(),
    }

    print(json.dumps(metrics, indent=2))
    model.save_model(args.model_out)


if __name__ == "__main__":
    main()
