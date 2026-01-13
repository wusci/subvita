from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pyreadstat


def xpt_to_parquet(xpt_path: Path, out_path: Path) -> None:
    """Read a SAS XPT file (NHANES) and write to Parquet."""
    df, meta = pyreadstat.read_xport(str(xpt_path))

    # Basic sanity checks
    if "SEQN" not in df.columns:
        raise ValueError(f"{xpt_path.name}: missing SEQN column; cannot merge later.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use pyarrow engine for consistent parquet writing
    df.to_parquet(out_path, engine="pyarrow", index=False)


def main() -> int:
    cycle = "2017-2018"
    raw_dir = Path("data_raw") / cycle
    out_dir = Path("data_interim") / cycle

    if not raw_dir.exists():
        print(f"ERROR: raw_dir not found: {raw_dir}", file=sys.stderr)
        return 1

    xpt_files = sorted(raw_dir.glob("*.XPT"))
    if not xpt_files:
        print(f"ERROR: no .XPT files found in {raw_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(xpt_files)} XPT files in {raw_dir}")

    for xpt_path in xpt_files:
        out_path = out_dir / (xpt_path.stem.replace("_J", "") + ".parquet")
        print(f"Converting {xpt_path.name} -> {out_path}")

        try:
            xpt_to_parquet(xpt_path, out_path)
        except Exception as e:
            print(f"FAILED: {xpt_path.name}: {e}", file=sys.stderr)
            return 1

    print("Stage 2 complete: XPT -> Parquet conversion finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())