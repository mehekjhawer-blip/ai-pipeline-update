"""Step 3 - label creation: NORMAL / WARNING / CRITICAL from the health policy.

Each rule in ``config.THRESHOLDS`` yields a severity (0/1/2) per row; the row
label is the worst severity.  The same function is used to create training
labels and, at run time, as a deterministic safety net and a source of
human-readable reasons.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from . import config as C


def rule_severities(feats: pd.DataFrame) -> pd.DataFrame:
    """Per-rule severity (0/1/2) for every row; NaN features give 0."""
    out = {}
    for name, r in C.THRESHOLDS.items():
        x = feats[r["feature"]].to_numpy(float)
        with np.errstate(invalid="ignore"):
            if r["dir"] == "hi":
                sev = (x >= r["warn"]).astype(int) + (x >= r["crit"]).astype(int)
            else:
                sev = (x <= r["warn"]).astype(int) + (x <= r["crit"]).astype(int)
        sev[np.isnan(x)] = 0
        out[name] = sev
    return pd.DataFrame(out, index=feats.index)


def label_rows(feats: pd.DataFrame) -> pd.Series:
    sev = rule_severities(feats)
    return sev.max(axis=1).rename("label")


def explain(sev_row: pd.Series, feat_row: pd.Series) -> Tuple[List[str], str]:
    """Reasons (worst first) and the recommendation for the worst rule."""
    hits = sorted(((s, n) for n, s in sev_row.items() if s > 0),
                  key=lambda h: (-h[0], C.RULE_PRIORITY.index(h[1])))
    reasons, rec = [], ""
    for s, name in hits:
        msg, r = C.MESSAGES[name]
        reasons.append(msg.format(v=feat_row[C.THRESHOLDS[name]["feature"]]))
        rec = rec or r
    return reasons, rec
