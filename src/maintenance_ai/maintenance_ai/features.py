"""Step 2 - feature engineering.

One function, :func:`compute_features`, is used for BOTH offline training and
live inference (the live engine just calls it on its sliding buffer), so the
model never sees a feature at run time that it did not see in training.

Beyond rolling statistics, three *residual* features compare what the robot is
doing with what its load should cause (models learned from NORMAL data):

* ``cur_resid_5s``       motor current minus the current its speed explains
* ``thermal_resid_cps``  observed heating rate minus the rate I^2 heating / cooling explains
* ``volt_resid_vps``     observed voltage fall rate minus the rate its current explains
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import config as C

DERIVED = [
    "temp_above_ambient",
    "speed_mean_5s",
    "ang_mean_5s",
    "current_mean_5s",
    "current_std_5s",
    "current_max_5s",
    "temp_slope_cps",
    "volt_slope_vps",
    "cur_resid_5s",
    "thermal_resid_cps",
    "volt_resid_vps",
]
FEATURE_NAMES = C.RAW_COLUMNS + DERIVED
# The anomaly detector only looks at "does the behaviour match the load?" residuals,
# so unfamiliar-but-healthy driving styles or battery levels do not trigger it.
ANOMALY_FEATURES = ["cur_resid_5s", "thermal_resid_cps", "volt_resid_vps"]


@dataclass
class ResidualModels:
    """Tiny linear 'normal behaviour' models (coefficients only, easy to pickle)."""

    current: np.ndarray   # I ~ [1, linear_speed, angular_speed]
    thermal: np.ndarray   # dT/dt ~ [mean(I^2), mean(T-ambient), 1]
    voltage: np.ndarray   # dV/dt ~ [mean(I), 1]


def _lag_slope(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Average rate of change over the last WINDOW samples (or since segment start)."""
    n = len(values)
    lag = np.maximum(np.arange(n) - (C.WINDOW - 1), 0)
    dt = t - t[lag]
    dv = values - values[lag]
    out = np.zeros(n)
    ok = dt > 1e-6
    out[ok] = dv[ok] / dt[ok]
    out[: C.MIN_WINDOW - 1] = np.nan
    return out


def _segment_stats(seg: pd.DataFrame) -> pd.DataFrame:
    """Window statistics that do not need the residual models."""
    W, M = C.WINDOW, C.MIN_WINDOW
    cur = seg["motor_current"]
    t = seg["timestamp"].to_numpy(float)
    st = pd.DataFrame(index=seg.index)
    st["temp_above_ambient"] = seg["motor_temperature"] - C.SIM["ambient"]
    st["speed_mean_5s"] = seg["linear_speed"].rolling(W, min_periods=M).mean()
    st["ang_mean_5s"] = seg["angular_speed"].rolling(W, min_periods=M).mean()
    st["current_mean_5s"] = cur.rolling(W, min_periods=M).mean()
    st["current_std_5s"] = cur.rolling(W, min_periods=M).std(ddof=0)
    st["current_max_5s"] = cur.rolling(W, min_periods=M).max()
    st["temp_slope_cps"] = _lag_slope(seg["motor_temperature"].to_numpy(float), t)
    st["volt_slope_vps"] = _lag_slope(seg["battery_voltage"].to_numpy(float), t)
    st["_i2_mean"] = (cur ** 2).rolling(W, min_periods=M).mean()
    st["_tamb_mean"] = st["temp_above_ambient"].rolling(W, min_periods=M).mean()
    return st


def _per_segment(df: pd.DataFrame, fn) -> pd.DataFrame:
    if "segment" not in df.columns:
        df = df.assign(segment=0)
    parts = [fn(seg) for _, seg in df.groupby("segment", sort=False)]
    return pd.concat(parts).loc[df.index] if parts else pd.DataFrame(index=df.index)


def fit_residual_models(clean_df: pd.DataFrame) -> ResidualModels:
    """Learn normal behaviour from data that contains NO faults."""
    st = _per_segment(clean_df, _segment_stats)
    d = pd.concat([clean_df, st.drop(columns=[c for c in st if c in clean_df])], axis=1)

    A = np.c_[np.ones(len(d)), d["linear_speed"], d["angular_speed"]]
    cur = np.linalg.lstsq(A, d["motor_current"].to_numpy(float), rcond=None)[0]

    v = d.dropna(subset=["temp_slope_cps", "volt_slope_vps", "_i2_mean", "current_mean_5s"])
    X = np.c_[v["_i2_mean"], v["_tamb_mean"], np.ones(len(v))]
    therm = np.linalg.lstsq(X, v["temp_slope_cps"].to_numpy(float), rcond=None)[0]
    X = np.c_[v["current_mean_5s"], np.ones(len(v))]
    volt = np.linalg.lstsq(X, v["volt_slope_vps"].to_numpy(float), rcond=None)[0]
    return ResidualModels(current=cur, thermal=therm, voltage=volt)


def compute_features(df: pd.DataFrame, models: ResidualModels) -> pd.DataFrame:
    """Feature matrix (``FEATURE_NAMES`` columns), NaN during the warm-up rows."""

    def one(seg: pd.DataFrame) -> pd.DataFrame:
        st = _segment_stats(seg)
        expected = (models.current[0] + models.current[1] * seg["linear_speed"]
                    + models.current[2] * seg["angular_speed"])
        st["cur_resid_5s"] = (seg["motor_current"] - expected).rolling(
            C.WINDOW, min_periods=C.MIN_WINDOW).mean()
        pred_t = (models.thermal[0] * st["_i2_mean"] + models.thermal[1] * st["_tamb_mean"]
                  + models.thermal[2])
        st["thermal_resid_cps"] = st["temp_slope_cps"] - pred_t
        pred_v = models.voltage[0] * st["current_mean_5s"] + models.voltage[1]
        st["volt_resid_vps"] = st["volt_slope_vps"] - pred_v
        return pd.concat([seg[C.RAW_COLUMNS], st[DERIVED]], axis=1)

    feats = _per_segment(df, one)
    return feats[FEATURE_NAMES]
