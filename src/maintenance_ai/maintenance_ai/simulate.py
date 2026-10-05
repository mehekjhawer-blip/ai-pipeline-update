"""Synthetic telemetry that follows the equations of ``maintenance_sensor_sim.cpp``.

Why: the real CSV only contains a couple of minutes of healthy driving, so it
has no WARNING / CRITICAL examples and few battery states.  We therefore replay
the simulator's own physics in Python with random driving, then overlay the
faults from :mod:`maintenance_ai.faults`.
"""

from typing import Optional

import numpy as np
import pandas as pd

from . import config as C
from .faults import FaultInjector, FaultProfile

LINEAR_CHOICES = [0.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.5, 0.8, 1.0]
ANGULAR_CHOICES = [0.0, 0.0, 0.0, 0.3, 0.8, 0.8, 1.5, 2.5]
MAX_NORMAL_LOAD = 9.0  # A of motion load allowed in healthy driving


def _random_command(rng: np.random.Generator):
    for _ in range(20):
        lin, ang = rng.choice(LINEAR_CHOICES), rng.choice(ANGULAR_CHOICES)
        if C.SIM["lin_gain"] * lin + C.SIM["ang_gain"] * ang <= MAX_NORMAL_LOAD:
            return float(lin), float(ang)
    return 0.0, 0.0


def simulate_session(
    rng: np.random.Generator,
    duration_s: float,
    fault: Optional[FaultProfile] = None,
    fault_onset_s: float = 30.0,
    start_voltage: Optional[float] = None,
    start_temp: Optional[float] = None,
    t0: float = 1.7e9,
) -> pd.DataFrame:
    """Return a 5 Hz session shaped like the logger CSV (+ ``fault_active`` flag)."""
    S = C.SIM
    v = start_voltage if start_voltage is not None else (
        rng.uniform(22.0, 24.0) if rng.random() < 0.85 else rng.uniform(18.5, 22.0))
    temp = start_temp if start_temp is not None else rng.uniform(28.0, 36.0)
    inj = FaultInjector()
    cmd_lin = cmd_ang = 0.0
    act_lin = act_ang = 0.0
    hold = 0.0
    rows, prev_speeds = [], (0.0, 0.0)
    ticks = int(duration_s / S["tick"])
    for k in range(ticks):
        now = k * S["tick"]
        if hold <= 0:                                  # new "keyboard" command
            cmd_lin, cmd_ang = _random_command(rng)
            hold = rng.uniform(0.4, 5.0)          # keyboard taps are short
        hold -= S["tick"]
        alpha = 1 - np.exp(-S["tick"] / 0.03)          # Gazebo odometry is almost step-like
        act_lin += alpha * (cmd_lin - act_lin)
        act_ang += alpha * (cmd_ang - act_ang)
        lin = max(act_lin + rng.normal(0, 0.003), 0.0) if cmd_lin or act_lin > 0.01 else 0.0
        ang = max(act_ang + rng.normal(0, 0.003), 0.0) if cmd_ang or act_ang > 0.01 else 0.0

        cur = min(S["idle_current"] + S["lin_gain"] * lin + S["ang_gain"] * ang, S["max_current"])
        drop = S["drop_idle"] if cur <= 0.6 else S["drop_per_amp"] * cur
        v = max(v - drop, S["v_empty"])
        temp += (cur ** 2 * S["heat_coeff"] - (temp - S["ambient"]) * S["cool_coeff"]) * S["tick"]
        temp = float(np.clip(temp, S["ambient"], S["max_temp"]))

        if k % 2 == 1:                                 # logger samples at 5 Hz
            speeds = prev_speeds if rng.random() < 0.6 else (lin, ang)   # 1-tick skew between speed and current, as in the real log
            row = {
                "linear_speed": speeds[0], "angular_speed": speeds[1],
                "motor_current": cur, "motor_temperature": temp, "battery_voltage": v,
                "battery_soc": float(np.clip((v - S["v_empty"]) / (S["v_full"] - S["v_empty"]) * 100, 0, 100)),
            }
            if fault is not None and now >= fault_onset_s:
                if inj.profile is None:
                    inj.set_fault(fault)
                row = inj.apply(row)
            row["timestamp"] = t0 + now
            row["fault_active"] = int(fault is not None and now >= fault_onset_s)
            rows.append(row)
        prev_speeds = (lin, ang)
    df = pd.DataFrame(rows)
    return df[["timestamp"] + C.RAW_COLUMNS + ["fault_active"]]


def build_dataset(seed: int = 7, n_normal: int = 70, n_per_fault: int = 30,
                  n_stuck: int = 20) -> pd.DataFrame:
    """Many sessions concatenated.  Columns: CSV columns + session, fault_kind, fault_active."""
    rng = np.random.default_rng(seed)
    sessions, sid = [], 0

    def add(df, kind):
        nonlocal sid
        df = df.copy()
        df["session"] = sid
        df["fault_kind"] = kind
        df["timestamp"] += sid * 10_000.0              # keep sessions apart in time
        sessions.append(df)
        sid += 1

    for _ in range(n_normal):
        add(simulate_session(rng, rng.uniform(120, 420)), "none")
    for kind in ("overheat", "drag", "battery"):
        for _ in range(n_per_fault):
            dur = rng.uniform(150, 420)
            add(simulate_session(rng, dur, FaultProfile.random(kind, rng),
                                 fault_onset_s=rng.uniform(20, 0.5 * dur)), kind)
    for _ in range(n_stuck):
        dur = rng.uniform(150, 300)
        add(simulate_session(rng, dur, FaultProfile.random("stuck_current", rng),
                             fault_onset_s=rng.uniform(20, 0.4 * dur)), "stuck_current")
    return pd.concat(sessions, ignore_index=True)
