from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


CYCLE = "2017-2018"
SPLIT_DIR = Path("data_processed") / CYCLE / "splits"
MODEL_DIR = Path("data_processed") / CYCLE / "models"
REPORT_DIR = Path("data_processed") / CYCLE / "reports"

RANDOM_SEED = 42


def load_split(split: str):
    X = pd.read_parquet(SPLIT_DIR / f"X_{split}.parquet")
    y = pd.read_parquet(SPLIT_DIR / f"y_{split}.parquet")["label_t2d_status"].astype(int)
    return X, y


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, y_train = load_split("train")
    X_valid, y_valid = load_split("valid")
    X_test, y_test = load_split("test")

    # Load feature list (optional safety check)
    fl_path = SPLIT_DIR / "feature_list.json"
    if fl_path.exists():
        feature_list = json.loads(fl_path.read_text(encoding="utf-8"))
        # Ensure column order matches frozen list
        X_train = X_train[feature_list]
        X_valid = X_valid[feature_list]
        X_test = X_test[feature_list]
    else:
        feature_list = list(X_train.columns)

    # Identify categorical vs numeric columns
    categorical_cols = [c for c in feature_list if X_train[c].dtype == "object"]
    numeric_cols = [c for c in feature_list if c not in categorical_cols]

    from sklearn.preprocessing import StandardScaler

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Baseline model: multinomial logistic regression
    base_clf = LogisticRegression(
    	solver="lbfgs",
    	max_iter=5000,
    	random_state=RANDOM_SEED,
)


    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", base_clf),
        ]
    )

    # Train base model
    pipeline.fit(X_train, y_train)

    # Evaluate base model on valid and test
    def evaluate(model, X, y, name: str) -> str:
        pred = model.predict(X)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred, labels=[0, 1, 2])
        rep = classification_report(y, pred, labels=[0, 1, 2], target_names=["normal", "prediabetes", "diabetes"])
        out = []
        out.append(f"== {name} ==")
        out.append(f"Accuracy: {acc:.4f}")
        out.append("Confusion matrix (rows=true, cols=pred) labels=[0,1,2]:")
        out.append(str(cm))
        out.append("\nClassification report:\n" + rep)
        return "\n".join(out)

    base_valid_report = evaluate(pipeline, X_valid, y_valid, "BASE (uncalibrated) - VALID")
    base_test_report = evaluate(pipeline, X_test, y_test, "BASE (uncalibrated) - TEST")

    (REPORT_DIR / "stage7_base_valid.txt").write_text(base_valid_report, encoding="utf-8")
    (REPORT_DIR / "stage7_base_test.txt").write_text(base_test_report, encoding="utf-8")

    print(base_valid_report)
    print("\n" + base_test_report)

    # Probability calibration using CV on TRAIN (sklearn >= 1.8 compatible)
    # This avoids the removed cv="prefit" behavior.
    calibrated = CalibratedClassifierCV(estimator=pipeline, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)


    cal_valid_report = evaluate(calibrated, X_valid, y_valid, "CALIBRATED (isotonic) - VALID")
    cal_test_report = evaluate(calibrated, X_test, y_test, "CALIBRATED (isotonic) - TEST")

    (REPORT_DIR / "stage7_calibrated_valid.txt").write_text(cal_valid_report, encoding="utf-8")
    (REPORT_DIR / "stage7_calibrated_test.txt").write_text(cal_test_report, encoding="utf-8")

    print("\n" + cal_valid_report)
    print("\n" + cal_test_report)

    # Save predicted probabilities on test (for later risk UI work)
    proba_test = calibrated.predict_proba(X_test)
    proba_df = pd.DataFrame(proba_test, columns=["p_normal", "p_prediabetes", "p_diabetes"])
    proba_df["y_true"] = y_test.values
    proba_df.to_parquet(REPORT_DIR / "stage7_test_probabilities.parquet", engine="pyarrow", index=False)

    # Save model artifacts
    joblib.dump(calibrated, MODEL_DIR / "model_a_calibrated.joblib")
    (MODEL_DIR / "feature_list.json").write_text(json.dumps(feature_list, indent=2), encoding="utf-8")

    print(f"\nSaved calibrated model to: {MODEL_DIR / 'model_a_calibrated.joblib'}")
    print(f"Saved reports to: {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())