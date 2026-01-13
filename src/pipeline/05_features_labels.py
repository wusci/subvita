from __future__ import annotations

from pathlib import Path
import pandas as pd


CYCLE = "2017-2018"
IN_PATH = Path("data_processed") / CYCLE / "model_a_merged.parquet"
OUT_PATH = Path("data_processed") / CYCLE / "model_a_features_labels.parquet"


def main() -> int:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing Stage 4 output: {IN_PATH}")

    df = pd.read_parquet(IN_PATH)

    # ----------------------------
    # 1) Enforce "strict cohort" for modeling
    #    (fasting subsample + required fasting lab values present)
    # ----------------------------
    required_for_strict = [
        "fasting_glucose_mg_dL",
        "triglycerides_mg_dL",
        "hdl_mg_dL",
        "age_years",
        "sex_at_birth",
        "waist_circumference_cm",
        "systolic_bp_mmHg",
        "diastolic_bp_mmHg",
    ]

    # HbA1c is extremely useful but not always present; we won't require it.
    # If you want to require it too, add "hba1c_percent" to required_for_strict.
    before = len(df)
    df = df.dropna(subset=required_for_strict).copy()
    after = len(df)

    print(f"Strict cohort filtering: {before} -> {after} rows (dropped {before-after})")

    # ----------------------------
    # 2) Feature engineering (safe derived features)
    # ----------------------------
    # BMI: prefer existing bmi if present; else compute from height/weight.
    if "bmi" not in df.columns:
        df["bmi"] = pd.NA

    if df["bmi"].isna().any():
        if "height_cm" in df.columns and "weight_kg" in df.columns:
            h_m = df["height_cm"] / 100.0
            df.loc[df["bmi"].isna() & h_m.notna() & df["weight_kg"].notna(), "bmi"] = (
                df["weight_kg"] / (h_m * h_m)
            )

    # TG/HDL ratio
    df["tg_to_hdl_ratio"] = df["triglycerides_mg_dL"] / df["hdl_mg_dL"]

    # Non-HDL cholesterol if total cholesterol present
    if "total_cholesterol_mg_dL" in df.columns and df["total_cholesterol_mg_dL"].notna().any():
        df["non_hdl_chol_mg_dL"] = df["total_cholesterol_mg_dL"] - df["hdl_mg_dL"]
    else:
        df["non_hdl_chol_mg_dL"] = pd.NA

    # Helpful combined meds flag (for reporting + labels)
    df["on_glucose_lowering_meds"] = (
        (df.get("on_insulin_now") == "yes") | (df.get("on_diabetes_pills_now") == "yes")
    ).map({True: "yes", False: "no"})

    # ----------------------------
    # 3) Build Model A labels
    # ----------------------------
    a1c = df.get("hba1c_percent")
    glu = df.get("fasting_glucose_mg_dL")

    diabetes_by_a1c = a1c.notna() & (a1c >= 6.5)
    diabetes_by_glu = glu.notna() & (glu >= 126)
    diabetes_by_selfreport = df.get("diabetes_self_report", pd.Series(["unknown"] * len(df))) == "yes"
    diabetes_by_meds = df["on_glucose_lowering_meds"] == "yes"

    diabetes = diabetes_by_a1c | diabetes_by_glu | diabetes_by_selfreport | diabetes_by_meds

    prediabetes_by_a1c = a1c.notna() & (a1c >= 5.7) & (a1c < 6.5)
    prediabetes_by_glu = glu.notna() & (glu >= 100) & (glu < 126)
    prediabetes = (~diabetes) & (prediabetes_by_a1c | prediabetes_by_glu)

    # label: 0 normal, 1 prediabetes, 2 diabetes
    df["label_t2d_status"] = 0
    df.loc[prediabetes, "label_t2d_status"] = 1
    df.loc[diabetes, "label_t2d_status"] = 2

    # Debug flags (very useful)
    df["diabetes_by_a1c"] = diabetes_by_a1c
    df["diabetes_by_glucose"] = diabetes_by_glu
    df["diabetes_by_selfreport"] = diabetes_by_selfreport
    df["diabetes_by_meds"] = diabetes_by_meds
    df["prediabetes_by_a1c"] = prediabetes_by_a1c
    df["prediabetes_by_glucose"] = prediabetes_by_glu

    # ----------------------------
    # 4) Basic reporting
    # ----------------------------
    counts = df["label_t2d_status"].value_counts().sort_index()
    label_names = {0: "normal", 1: "prediabetes", 2: "diabetes"}

    print("\nLabel counts:")
    for k, v in counts.items():
        print(f"  {k} ({label_names.get(k,'?')}): {v}")

    # Missingness snapshot for key fields
    key_cols = [
        "hba1c_percent",
        "fasting_glucose_mg_dL",
        "triglycerides_mg_dL",
        "hdl_mg_dL",
        "waist_circumference_cm",
        "systolic_bp_mmHg",
        "diastolic_bp_mmHg",
        "bmi",
        "alt_U_L",
        "creatinine_mg_dL",
        "total_cholesterol_mg_dL",
    ]
    print("\nMissingness (key columns):")
    for c in key_cols:
        if c in df.columns:
            print(f"  {c:28s} {df[c].isna().mean():.3f}")

    # ----------------------------
    # 5) Write output
    # ----------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    print(f"\nWrote {OUT_PATH} rows={len(df)} cols={len(df.columns)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())