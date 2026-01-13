import pandas as pd

df = pd.read_parquet("data_processed/2017-2018/model_a_merged.parquet")
print("Rows:", len(df), "Cols:", len(df.columns))
print("\nColumns:")
print(sorted(df.columns))
print("\nHead:")
print(df.head(3).to_string(index=False))