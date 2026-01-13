from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


CYCLE = "2017-2018"
IN_PATH = Path("data_processed") / CYCLE / "model_a_features_labels.parquet"
OUT_DIR = Path("data_processed") / CYCLE / "splits"

RANDOM_SEED = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15  # of total dataset (not of remaining)


LEAKAGE_COLS = [
    # label itself handled separately
    "diabetes_by_a1c",
    "diabetes_by_glucose",
    "diabetes_by_selfreport",
    "diabetes_by_meds",
    "prediabetes_by_a1c",
    "prediabetes_by_glucose",
    # raw label-definition fields (avoid leakage)
    "diabetes_self_report",
    "on_insulin_now",
    "on_diabetes_pills_now",
    "on_glucose_lowering_meds",
]


META_COLS = ["SEQN"]


def main() -> int:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PATH)

    if "label_t2d_status" not in df.columns:
        raise ValueError("label_t2d_status not found in dataset.")

    y = df["label_t2d_status"].astype(int)

    # Build feature frame
    drop_cols = set(LEAKAGE_COLS + META_COLS + ["label_t2d_status"])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    seqn = df["SEQN"].copy() if "SEQN" in df.columns else None

    # First split: train+valid vs test
    X_trainvalid, X_test, y_trainvalid, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Second split: train vs valid
    # valid should be VALID_SIZE of total, which is VALID_SIZE/(1-TEST_SIZE) of trainvalid
    valid_fraction_of_trainvalid = VALID_SIZE / (1.0 - TEST_SIZE)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainvalid, y_trainvalid,
        test_size=valid_fraction_of_trainvalid,
        random_state=RANDOM_SEED,
        stratify=y_trainvalid
    )

    # Save outputs
    def save_pair(split_name: str, X_split: pd.DataFrame, y_split: pd.Series) -> None:
        X_split.to_parquet(OUT_DIR / f"X_{split_name}.parquet", engine="pyarrow", index=False)
        y_split.to_frame("label_t2d_status").to_parquet(OUT_DIR / f"y_{split_name}.parquet", engine="pyarrow", index=False)

    save_pair("train", X_train, y_train)
    save_pair("valid", X_valid, y_valid)
    save_pair("test", X_test, y_test)

    # Save feature list
    with (OUT_DIR / "feature_list.json").open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # Print summary
    def summarize(name: str, ys: pd.Series) -> None:
        vc = ys.value_counts().sort_index()
        total = len(ys)
        print(f"\n{name} n={total}")
        for k, v in vc.items():
            print(f"  label {k}: {v} ({v/total:.3f})")

    print(f"Saved splits to: {OUT_DIR}")
    print(f"Num features: {len(feature_cols)}")

    summarize("TRAIN", y_train)
    summarize("VALID", y_valid)
    summarize("TEST", y_test)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())