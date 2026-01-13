from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    log_loss,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance


CYCLE = "2017-2018"
SPLIT_DIR = Path("data_processed") / CYCLE / "splits"
MODEL_DIR = Path("data_processed") / CYCLE / "models"
REPORT_DIR = Path("data_processed") / CYCLE / "reports"

MODEL_PATH = MODEL_DIR / "model_a_calibrated.joblib"


LABELS = [0, 1, 2]
LABEL_NAMES = {0: "normal", 1: "prediabetes", 2: "diabetes"}


def load_split(split: str):
    X = pd.read_parquet(SPLIT_DIR / f"X_{split}.parquet")
    y = pd.read_parquet(SPLIT_DIR / f"y_{split}.parquet")["label_t2d_status"].astype(int)
    return X, y


def save_confusion_matrix(y_true, y_pred, title: str, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    disp = ConfusionMatrixDisplay(cm, display_labels=[LABEL_NAMES[i] for i in LABELS])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def multiclass_calibration_plot(y_true, proba, out_path: Path, n_bins: int = 10):
    """
    One-vs-rest calibration curves for each class.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for k in LABELS:
        y_bin = (y_true == k).astype(int)
        frac_pos, mean_pred = calibration_curve(y_bin, proba[:, k], n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1, label=LABEL_NAMES[k])

    # reference line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curves (one-vs-rest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # Load splits
    X_train, y_train = load_split("train")
    X_valid, y_valid = load_split("valid")
    X_test, y_test = load_split("test")

    # Predict
    y_pred_test = model.predict(X_test)
    proba_test = model.predict_proba(X_test)

    # ---- 1) Confusion matrix plot ----
    save_confusion_matrix(
        y_test,
        y_pred_test,
        title="Confusion Matrix (TEST) — Calibrated model",
        out_path=REPORT_DIR / "stage8_confusion_matrix_test.png",
    )

    # ---- 2) Text classification report ----
    rep = classification_report(
        y_test, y_pred_test,
        labels=LABELS,
        target_names=[LABEL_NAMES[i] for i in LABELS]
    )
    (REPORT_DIR / "stage8_classification_report_test.txt").write_text(rep, encoding="utf-8")

    # ---- 3) Calibration metrics ----
    # Log loss (multiclass)
    ll = log_loss(y_test, proba_test, labels=LABELS)

    # Brier score (one-vs-rest per class)
    brier = {}
    for k in LABELS:
        y_bin = (y_test == k).astype(int)
        brier[LABEL_NAMES[k]] = float(brier_score_loss(y_bin, proba_test[:, k]))

    cal_summary = {
        "log_loss_multiclass": float(ll),
        "brier_ovr": brier,
    }
    (REPORT_DIR / "stage8_calibration_metrics.json").write_text(
        json.dumps(cal_summary, indent=2),
        encoding="utf-8"
    )

    # Calibration curve plot
    multiclass_calibration_plot(
        y_test.values,
        proba_test,
        out_path=REPORT_DIR / "stage8_calibration_curves_test.png",
        n_bins=10
    )

    # ---- 4) ROC-AUC (one-vs-rest) ----
    # Binarize labels for OVR ROC-AUC
    y_test_bin = label_binarize(y_test, classes=LABELS)
    # roc_auc_score expects shape (n_samples, n_classes)
    auc_ovr = roc_auc_score(y_test_bin, proba_test, average=None)
    auc_macro = roc_auc_score(y_test_bin, proba_test, average="macro")
    auc_weighted = roc_auc_score(y_test_bin, proba_test, average="weighted")

    auc_summary = {
        "auc_ovr": {LABEL_NAMES[k]: float(auc_ovr[i]) for i, k in enumerate(LABELS)},
        "auc_macro": float(auc_macro),
        "auc_weighted": float(auc_weighted),
    }
    (REPORT_DIR / "stage8_auc_ovr.json").write_text(json.dumps(auc_summary, indent=2), encoding="utf-8")

    # ---- 5) Permutation importance (model-agnostic) ----
    # This is safe even with one-hot encoding & calibration, because it measures drop in score.
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring="accuracy",
        n_jobs=-1
    )

    importances = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    importances.to_csv(REPORT_DIR / "stage8_permutation_importance_test.csv", index=False)

    # Plot top 15 importances
    topn = importances.head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(topn["feature"][::-1], topn["importance_mean"][::-1])
    ax.set_xlabel("Decrease in accuracy (permutation importance)")
    ax.set_title("Top 15 permutation importances (TEST)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "stage8_permutation_importance_test.png", dpi=160)
    plt.close(fig)

    # ---- 6) Save test probabilities with IDs (useful for app QA later) ----
    proba_df = pd.DataFrame(proba_test, columns=["p_normal", "p_prediabetes", "p_diabetes"])
    proba_df["y_true"] = y_test.values
    proba_df["y_pred"] = y_pred_test
    # keep an index for traceability
    proba_df.to_parquet(REPORT_DIR / "stage8_test_predictions.parquet", engine="pyarrow", index=False)

    print("Stage 8 complete. Artifacts written to:")
    print(REPORT_DIR)

    print("\nKey metrics (TEST):")
    print(f"  log_loss (multiclass): {ll:.4f}")
    for k, v in brier.items():
        print(f"  brier ({k}): {v:.4f}")
    print("  AUC OVR:", auc_summary["auc_ovr"])
    print(f"  AUC macro: {auc_macro:.4f}  weighted: {auc_weighted:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
