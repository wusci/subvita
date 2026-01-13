from __future__ import annotations

from pathlib import Path
import sys
import math

import pandas as pd

try:
    import yaml  # pyyaml
except ImportError:
    yaml = None


MISSING_SENTINELS = {7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999}


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("Missing dependency: pyyaml. Install with: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_null_if_sentinel(series: pd.Series) -> pd.Series:
    # Replace common NHANES missing codes with NA
    if pd.api.types.is_numeric_dtype(series):
        return series.where(~series.isin(MISSING_SENTINELS), pd.NA)
    return series


def map_sex(riagendr: pd.Series) -> pd.Series:
    # NHANES RIAGENDR: 1=Male, 2=Female (typically)
    mapped = riagendr.map({1: "male", 2: "female"})
    return mapped.fillna("unknown")


def map_race_ethnicity(ridreth: pd.Series) -> pd.Series:
    # RIDRETH3 common mapping:
    # 1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic White,
    # 4=Non-Hispanic Black, 6=Non-Hispanic Asian, 7=Other/Multi
    mapping = {
        1: "mexican_american",
        2: "other_hispanic",
        3: "non_hispanic_white",
        4: "non_hispanic_black",
        6: "non_hispanic_asian",
        7: "other_or_multiracial",
    }
    mapped = ridreth.map(mapping)
    return mapped.fillna("unknown")


def map_pregnancy(ridexprg: pd.Series) -> pd.Series:
    # RIDEXPRG often: 1=Pregnant, 2=Not pregnant, 3=Unknown
    mapping = {1: "pregnant", 2: "not_pregnant"}
    mapped = ridexprg.map(mapping)
    return mapped.fillna("unknown")


def map_yes_no_unknown(series: pd.Series) -> pd.Series:
    # Many NHANES yes/no items: 1=Yes, 2=No, 7/9 missing
    series = to_null_if_sentinel(series)
    mapped = series.map({1: "yes", 2: "no"})
    return mapped.fillna("unknown")


def mean_of_readings(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    # Take mean across non-null readings
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series([pd.NA] * len(df), index=df.index)
    sub = df[existing].apply(to_null_if_sentinel)
    return sub.mean(axis=1, skipna=True)


def standardize_table(
    df: pd.DataFrame,
    seqn_col: str,
    fields: dict[str, str],
) -> pd.DataFrame:
    out = pd.DataFrame()
    out["SEQN"] = df[seqn_col]

    for canon, raw in fields.items():
        if raw not in df.columns:
            out[canon] = pd.NA
            continue
        out[canon] = to_null_if_sentinel(df[raw])

    return out


def main() -> int:
    cycle = "2017-2018"
    interim_dir = Path("data_interim") / cycle
    out_dir = Path("data_processed") / cycle / "standardized"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path("src/config/nhanes_2017_2018_map.yaml")
    cfg = load_yaml(cfg_path)

    # Load required parquet tables
    def load_table(name: str) -> pd.DataFrame:
        path = interim_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing interim parquet: {path}")
        return pd.read_parquet(path)

    demo = load_table("DEMO")
    bmx = load_table("BMX")
    bpx = load_table("BPX")
    ghb = load_table("GHB")
    glu = load_table("GLU")
    hdl = load_table("HDL")
    trig = load_table("TRIGLY")
    tchol = load_table("TCHOL")
    biopro = load_table("BIOPRO")
    diq = load_table("DIQ")

    # DEMO: rename + recode
    demo_map = cfg["tables"]["DEMO"]["fields"]
    demo_seqn = cfg["tables"]["DEMO"]["seqn"]

    # Some cycles might not have RIDRETH3; fallback to RIDRETH1 if needed
    if demo_map.get("race_ethnicity") not in demo.columns and "RIDRETH1" in demo.columns:
        demo_map = dict(demo_map)
        demo_map["race_ethnicity"] = "RIDRETH1"

    demo_std = standardize_table(demo, demo_seqn, demo_map)
    demo_std["sex_at_birth"] = map_sex(to_null_if_sentinel(demo[demo_map["sex_at_birth"]]))
    demo_std["race_ethnicity"] = map_race_ethnicity(to_null_if_sentinel(demo[demo_map["race_ethnicity"]]))
    demo_std["pregnancy_status"] = map_pregnancy(to_null_if_sentinel(demo.get(demo_map.get("pregnancy_status", ""), pd.Series([pd.NA]*len(demo)))))

    # BMX
    bmx_std = standardize_table(bmx, cfg["tables"]["BMX"]["seqn"], cfg["tables"]["BMX"]["fields"])

    # BPX: average readings
    bpx_std = pd.DataFrame({"SEQN": bpx["SEQN"]})
    bpx_std["systolic_bp_mmHg"] = mean_of_readings(bpx, cfg["tables"]["BPX"]["systolic_readings"])
    bpx_std["diastolic_bp_mmHg"] = mean_of_readings(bpx, cfg["tables"]["BPX"]["diastolic_readings"])

    # Labs
    ghb_std = standardize_table(ghb, cfg["tables"]["GHB"]["seqn"], cfg["tables"]["GHB"]["fields"])
    glu_std = standardize_table(glu, cfg["tables"]["GLU"]["seqn"], cfg["tables"]["GLU"]["fields"])
    hdl_std = standardize_table(hdl, cfg["tables"]["HDL"]["seqn"], cfg["tables"]["HDL"]["fields"])
    trig_std = standardize_table(trig, cfg["tables"]["TRIGLY"]["seqn"], cfg["tables"]["TRIGLY"]["fields"])
    tchol_std = standardize_table(tchol, cfg["tables"]["TCHOL"]["seqn"], cfg["tables"]["TCHOL"]["fields"])
    biopro_std = standardize_table(biopro, cfg["tables"]["BIOPRO"]["seqn"], cfg["tables"]["BIOPRO"]["fields"])

    # DIQ: keep raw flags + map to yes/no/unknown
    diq_fields = cfg["tables"]["DIQ"]["fields"]
    diq_std = standardize_table(diq, cfg["tables"]["DIQ"]["seqn"], diq_fields)

    # diabetes_self_report: DIQ010 often 1=Yes, 2=No
    if "diabetes_self_report" in diq_std.columns and diq_fields["diabetes_self_report"] in diq.columns:
        diq_std["diabetes_self_report"] = map_yes_no_unknown(diq[diq_fields["diabetes_self_report"]])

    # meds flags
    if "on_insulin_now" in diq_std.columns and diq_fields["on_insulin_now"] in diq.columns:
        diq_std["on_insulin_now"] = map_yes_no_unknown(diq[diq_fields["on_insulin_now"]])

    if "on_diabetes_pills_now" in diq_std.columns and diq_fields["on_diabetes_pills_now"] in diq.columns:
        diq_std["on_diabetes_pills_now"] = map_yes_no_unknown(diq[diq_fields["on_diabetes_pills_now"]])

    # Write outputs
    outputs = {
        "DEMO_std.parquet": demo_std,
        "BMX_std.parquet": bmx_std,
        "BPX_std.parquet": bpx_std,
        "GHB_std.parquet": ghb_std,
        "GLU_std.parquet": glu_std,
        "HDL_std.parquet": hdl_std,
        "TRIGLY_std.parquet": trig_std,
        "TCHOL_std.parquet": tchol_std,
        "BIOPRO_std.parquet": biopro_std,
        "DIQ_std.parquet": diq_std,
    }

    for name, df in outputs.items():
        out_path = out_dir / name
        df.to_parquet(out_path, engine="pyarrow", index=False)
        print(f"Wrote {out_path} rows={len(df)} cols={len(df.columns)}")

    print("Stage 3 complete: standardized tables created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())