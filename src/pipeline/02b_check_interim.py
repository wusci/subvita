from pathlib import Path
import pandas as pd

cycle = "2017-2018"
p = Path("data_interim") / cycle

for f in sorted(p.glob("*.parquet")):
    df = pd.read_parquet(f)
    n = len(df)
    n_seqn = df["SEQN"].nunique() if "SEQN" in df.columns else 0
    print(f"{f.name:15s} rows={n:6d} unique_SEQN={n_seqn:6d} seqn_ok={n==n_seqn}")