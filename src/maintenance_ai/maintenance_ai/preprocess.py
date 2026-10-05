"""Step 1 - data cleaning and preprocessing.

The logger *appends* to ``maintenance_data.csv`` on every run, so the file can
hold several sessions glued together (battery back at 24 V, time gaps).  This
module validates the file, removes bad rows and tags each continuous run with
a ``segment`` id so rolling features never straddle two runs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    dropped: Dict[str, int] = field(default_factory=dict)
    segments: int = 0
    median_dt: float = float("nan")

    def __str__(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in self.dropped.items() if v) or "nothing dropped"
        return (f"cleaned {self.rows_in} -> {self.rows_out} rows ({parts}); "
                f"{self.segments} segment(s); median dt={self.median_dt:.3f}s")


def clean(df: pd.DataFrame):
    """Return ``(clean_df, report)``.  ``clean_df`` has a ``segment`` column."""
    missing = [c for c in C.CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))

    rep = CleaningReport(rows_in=len(df))
    out = df[C.CSV_COLUMNS].copy()

    for col in C.CSV_COLUMNS:                       # numeric coercion (repeated headers, text)
        out[col] = pd.to_numeric(out[col], errors="coerce")

    n = len(out)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    rep.dropped["non_numeric_or_nan"] = n - len(out)

    n = len(out)
    ok = pd.Series(True, index=out.index)
    for col, (lo, hi) in C.PLAUSIBLE.items():
        ok &= out[col].between(lo, hi)
    out = out[ok]
    rep.dropped["out_of_range"] = n - len(out)

    out = out.sort_values("timestamp", kind="stable")
    n = len(out)
    out = out.drop_duplicates(subset="timestamp", keep="first")
    rep.dropped["duplicate_timestamp"] = n - len(out)

    out = out.reset_index(drop=True)
    dt = out["timestamp"].diff()
    new_seg = (dt > C.GAP_SECONDS) | (out["battery_voltage"].diff() > C.VOLTAGE_RESET_V)
    out["segment"] = new_seg.cumsum().astype(int)

    rep.rows_out = len(out)
    rep.segments = int(out["segment"].nunique()) if len(out) else 0
    rep.median_dt = float(dt.median()) if len(out) > 1 else float("nan")
    return out, rep


def load_and_clean(path) -> tuple:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    return clean(pd.read_csv(path, encoding="utf-8-sig"))
