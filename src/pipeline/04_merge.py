from __future__ import annotations

from pathlib import Path
import pandas as pd


CYCLE = "2017-2018"
STD_DIR = Path("data_processed") / CYCLE / "standardized"
OUT_PATH = Path("data_processed") / CYCLE / "model_a_merged.parquet"


def read_std(name: str) -> pd.DataFrame:
    path = STD_DIR / f"{name}_std.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing standardized file: {path}")
    return pd.read_parquet(path)


def left_join(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    # Ensure no duplicate SEQN in incoming table
    if other["SEQN"].nunique() != len(other):
        raise ValueError("Incoming table has duplicate SEQN rows; must aggregate before merge.")
    return base.merge(other, on="SEQN", how="left")


def main() -> int:
    # Load standardized component tables
    demo = read_std("DEMO")
    bmx = read_std("BMX")
    bpx = read_std("BPX")
    ghb = read_std("GHB")
    glu = read_std("GLU")
    hdl = read_std("HDL")
    trig = read_std("TRIGLY")
    tchol = read_std("TCHOL")
    biopro = read_std("BIOPRO")
    diq = read_std("DIQ")

    # Choose your base cohort:
    # For "strict fasting cohort" (requires GLU/TRIGLY), start from GLU (or TRIGLY).
    base = glu.copy()

    # Merge everything onto base
    df = base
    for other in [demo, bmx, bpx, ghb, hdl, trig, tchol, biopro, diq]:
        df = left_join(df, other)

    # Create a couple useful derived features (optional but recommended)
    # BMI: if missing but height/weight exist, compute.
    if "bmi" in df.columns:
        pass
    else:
        df["bmi"] = pd.NA

    # TG/HDL ratio (only if both present and non-zero)
    if "triglycerides_mg_dL" in df.columns and "hdl_mg_dL" in df.columns:
        df["tg_to_hdl_ratio"] = df["triglycerides_mg_dL"] / df["hdl_mg_dL"]
    else:
        df["tg_to_hdl_ratio"] = pd.NA

    # Basic quality flags
    df["has_fasting_glucose"] = df["fasting_glucose_mg_dL"].notna()
    df["has_triglycerides"] = df["triglycerides_mg_dL"].notna()
    df["has_hba1c"] = df["hba1c_percent"].notna()

    # Write merged dataset
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, engine="pyarrow", index=False)

    print(f"Wrote {OUT_PATH}")
    print(f"Rows: {len(df)}  Cols: {len(df.columns)}")

    # Missingness report (top missing columns)
    miss = df.isna().mean().sort_values(ascending=False)
    top = miss.head(20)
    print("\nTop 20 missingness rates:")
    for col, rate in top.items():
        print(f"{col:30s} {rate:.3f}")

    # Quick sanity checks
    if df["SEQN"].nunique() != len(df):
        print("WARNING: duplicate SEQN found in merged table (unexpected).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())