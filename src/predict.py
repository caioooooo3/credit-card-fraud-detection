"""Train the final CatBoost model and generate a Kaggle submission."""

import argparse
import pandas as pd
from catboost import CatBoostClassifier

TARGET = "is_fraud"
RANDOM_STATE = 42
DEFAULT_ITERATIONS = 931


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["id"], errors="ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    submission = pd.read_csv(args.sample_submission)

    X = build_features(train.drop(columns=[TARGET]))
    y = train[TARGET].astype(int)
    X_test = build_features(test)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=0.05,
        depth=7,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=100,
    )

    model.fit(X, y, cat_features=cat_cols)

    submission[TARGET] = model.predict_proba(X_test[X.columns])[:, 1]
    submission.to_csv(args.output, index=False)

    print(f"Saved {args.output} with shape {submission.shape}")


if __name__ == "__main__":
    main()
