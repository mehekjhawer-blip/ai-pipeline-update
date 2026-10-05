"""Steps 7-8 - the inference engine (no ROS dependency, so it is easy to test).

``assess_frame``   batch assessment of a cleaned DataFrame (CSV / historical mode)
``HealthEngine``   streaming assessment, one telemetry row at a time (live mode)

Both share the same code path, so a recorded run and a live run give the same
answers.  Decision logic (``fuse``):

1. The classifier predicts NORMAL / WARNING / CRITICAL with a probability.
2. Deterministic policy rules are a safety net: final = max(classifier, rules).
3. If everything says NORMAL but the anomaly detector keeps flagging samples
   that look unlike healthy data, the status becomes WARNING (unknown fault).
"""

from collections import deque
from typing import Dict

import numpy as np
import pandas as pd

from . import config as C
from .assessment import WARMING_UP, Assessment  # noqa: F401  (re-exported)
from .features import ANOMALY_FEATURES
from .labeling import explain, rule_severities
from .model import HealthModel

def _persistent(flags: np.ndarray, segments: np.ndarray) -> np.ndarray:
    s = pd.Series(flags.astype(float))
    hits = s.groupby(segments).transform(
        lambda x: x.rolling(C.ANOMALY_WINDOW, min_periods=1).sum())
    return (hits >= C.ANOMALY_MIN_HITS).to_numpy()


def assess_frame(model: HealthModel, df: pd.DataFrame, last_only: bool = False,
                 explain_rows: bool = True) -> pd.DataFrame:
    """Assess every row of a cleaned frame (needs ``segment``)."""
    if "segment" not in df.columns:
        df = df.assign(segment=0)
    feats = model.features(df)
    valid = feats.notna().all(axis=1).to_numpy()
    n = len(df)

    scores = np.full(n, np.nan)
    if valid.any():
        scores[valid] = model.anomaly_scores(feats[valid])
    raw_flag = np.where(valid, scores < model.anomaly_threshold, False)
    anomaly = _persistent(raw_flag, df["segment"].to_numpy()) & valid

    want = valid.copy()
    if last_only:
        want[:-1] = False
    proba = np.full((n, 3), np.nan)
    if want.any():
        proba[want] = model.proba(feats[want])

    sev = rule_severities(feats)
    rule = sev.max(axis=1).to_numpy()
    ml = np.where(want, np.argmax(np.nan_to_num(proba, nan=-1.0), axis=1), 0)

    final = np.maximum(ml, rule)
    source = np.where(rule > ml, "safety rule", np.where(ml > 0, "classifier", ""))
    flip = (final == 0) & anomaly
    final = np.where(flip, 1, final)
    source = np.where(flip, "anomaly detector", source)

    conf = np.full(n, np.nan)
    for i in np.flatnonzero(want):
        if source[i] == "safety rule":
            conf[i] = 1.0
        elif source[i] == "anomaly detector":
            conf[i] = float(np.clip(0.5 + 4 * (model.anomaly_threshold - scores[i]), 0.5, 0.95))
        else:
            conf[i] = proba[i, final[i]]

    out = pd.DataFrame(index=df.index)
    out["status_idx"] = np.where(want, final, -1)
    out["status"] = [C.LABELS[k] if k >= 0 else WARMING_UP for k in out["status_idx"]]
    out["confidence"] = conf
    out["source"] = np.where(want, source, "")
    out["anomaly"] = anomaly
    out["anomaly_score"] = scores
    for j, lab in enumerate(C.LABELS):
        out[f"p_{lab.lower()}"] = proba[:, j]
    out["rule_idx"] = np.where(valid, rule, -1)
    out["ml_idx"] = np.where(want, ml, -1)

    reasons, recs = [""] * n, [""] * n
    if explain_rows:
        z_all, z_an = model.z_scores(feats), model.z_scores(feats, ANOMALY_FEATURES)
        for i in np.flatnonzero(want):
            rs, rec = explain(sev.iloc[i], feats.iloc[i])
            if not rs and final[i] > 0:
                zr = z_all.iloc[i] if source[i] == "classifier" else z_an.iloc[i]
                top = zr.abs().idxmax()
                lead = (f"Classifier sees a {C.LABELS[final[i]]} pattern"
                        if source[i] == "classifier" else C.ANOMALY_MESSAGE[0])
                rs = [f"{lead}; biggest deviation: {C.FEATURE_LABELS.get(top, top)} ({zr[top]:+.1f} sigma)"]
                rec = C.ANOMALY_MESSAGE[1]
            reasons[i] = "; ".join(rs) if rs else "All monitored values are within normal behaviour"
            recs[i] = rec
    out["reason"], out["recommendation"] = reasons, recs
    out.loc[~want, "reason"] = "Collecting data for the first window"
    return out


class HealthEngine:
    """Streaming wrapper: feed telemetry rows, get an :class:`Assessment` back."""

    BUFFER = C.WINDOW + C.ANOMALY_WINDOW - 1       # enough history for exact batch parity

    def __init__(self, model: HealthModel) -> None:
        self.model = model
        self.reset()

    def reset(self) -> None:
        self._rows: deque = deque(maxlen=self.BUFFER)

    def update(self, row: Dict[str, float], injected_fault: str = "none") -> Assessment:
        """``row`` needs ``timestamp`` (epoch s) and the six ``RAW_COLUMNS``."""
        if self._rows:
            last = self._rows[-1]
            if (row["timestamp"] - last["timestamp"] > C.GAP_SECONDS
                    or row["timestamp"] <= last["timestamp"]
                    or row["battery_voltage"] - last["battery_voltage"] > C.VOLTAGE_RESET_V):
                self.reset()
        self._rows.append({k: float(row[k]) for k in C.CSV_COLUMNS})
        df = pd.DataFrame(list(self._rows))
        df["segment"] = 0
        res = assess_frame(self.model, df, last_only=True).iloc[-1]
        a = Assessment(timestamp=float(row["timestamp"]), injected_fault=injected_fault)
        a.status = res["status"]
        if res["status_idx"] < 0:
            return a
        a.confidence = float(res["confidence"])
        a.reason, a.recommendation, a.source = res["reason"], res["recommendation"], res["source"]
        a.anomaly = bool(res["anomaly"])
        a.anomaly_score = float(res["anomaly_score"])
        a.probabilities = {lab: float(res[f"p_{lab.lower()}"]) for lab in C.LABELS}
        return a
