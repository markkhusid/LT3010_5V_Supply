"""Side-by-side analytical vs SPICE tables."""

from __future__ import annotations

import pandas as pd


def compare_table(rows: list[dict], *, cols: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if cols:
        keep = [c for c in cols if c in df.columns]
        df = df[keep]
    if "vout_an" in df.columns and "vout_spice" in df.columns:
        df["err_mV"] = 1e3 * (df["vout_spice"] - df["vout_an"])
    return df
