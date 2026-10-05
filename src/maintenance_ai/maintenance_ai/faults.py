"""Fault injection (used for synthetic training data AND the live demo).

A fault is applied *on top of* the healthy telemetry, so the same code path
creates training examples offline and demo faults on the real simulation,
without touching the existing C++ nodes.

Kinds
-----
overheat        cooling failure: motor temperature drifts up at a growing rate
drag            mechanical friction: extra current (which also heats the motor
                and drains the battery faster, like the real physics would)
battery         degraded battery: voltage falls faster than the load explains
stuck_current   UNSEEN fault: current sensor freezes (used to test the anomaly
                detector, which was never trained on it)
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from . import config as C

KINDS = ("overheat", "drag", "battery", "stuck_current")


@dataclass
class FaultProfile:
    kind: str
    # growth law: magnitude(t) = start + growth * t   (t = seconds since onset)
    start: float
    growth: float

    @staticmethod
    def demo(kind: str) -> "FaultProfile":
        return {
            "overheat": FaultProfile("overheat", 0.10, 0.010),        # C/s
            "drag": FaultProfile("drag", 0.30, 0.100),                # A
            "battery": FaultProfile("battery", 0.004, 0.0006),        # V/s
            "stuck_current": FaultProfile("stuck_current", 0.0, 0.0),
        }[kind]

    @staticmethod
    def random(kind: str, rng: np.random.Generator) -> "FaultProfile":
        if kind == "overheat":
            return FaultProfile(kind, rng.uniform(0.05, 0.30), rng.uniform(0.004, 0.025))
        if kind == "drag":
            return FaultProfile(kind, rng.uniform(0.2, 1.2), rng.uniform(0.04, 0.20))
        if kind == "battery":
            return FaultProfile(kind, rng.uniform(0.002, 0.010), rng.uniform(0.0003, 0.0012))
        return FaultProfile(kind, 0.0, 0.0)


class FaultInjector:
    """Stateful overlay.  ``apply`` takes one telemetry row and returns a faulted copy."""

    def __init__(self) -> None:
        self.profile: Optional[FaultProfile] = None
        self.reset()

    # -- control -----------------------------------------------------------
    def reset(self) -> None:
        self.t = 0.0
        self.extra_temp = 0.0
        self.volt_drop = 0.0
        self.frozen_current: Optional[float] = None

    def set_fault(self, profile: Optional[FaultProfile]) -> None:
        self.profile = profile
        self.reset()

    @property
    def active(self) -> str:
        return self.profile.kind if self.profile else "none"

    # -- overlay -----------------------------------------------------------
    def apply(self, row: Dict[str, float], dt: float = C.DT) -> Dict[str, float]:
        p = self.profile
        if p is None:
            return dict(row)
        S = C.SIM
        out = dict(row)
        mag = p.start + p.growth * self.t
        cur = row["motor_current"]

        if p.kind == "drag":
            new_cur = min(cur + mag, S["max_current"])
            heat_extra = (new_cur ** 2 - cur ** 2) * S["heat_coeff"]
            self.extra_temp += (heat_extra - S["cool_coeff"] * self.extra_temp) * dt
            if new_cur > 0.6:                         # extra drain, same law as the simulator
                self.volt_drop += (new_cur - cur) * S["drop_per_amp"] / S["tick"] * dt
            out["motor_current"] = new_cur
        elif p.kind == "overheat":
            self.extra_temp += mag * dt
        elif p.kind == "battery":
            self.volt_drop += mag * dt
        elif p.kind == "stuck_current":
            if self.frozen_current is None:
                self.frozen_current = cur
            out["motor_current"] = self.frozen_current

        out["motor_temperature"] = min(row["motor_temperature"] + self.extra_temp, S["max_temp"])
        v = max(row["battery_voltage"] - self.volt_drop, S["v_empty"])
        out["battery_voltage"] = v
        out["battery_soc"] = float(np.clip((v - S["v_empty"]) / (S["v_full"] - S["v_empty"]) * 100,
                                           0.0, 100.0))
        self.t += dt
        return out
