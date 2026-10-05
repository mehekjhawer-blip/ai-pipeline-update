"""Central configuration: sampling, windows, health thresholds, simulator constants.

Everything that a human might want to tune lives here.  The thresholds are an
engineering *policy* (what counts as WARNING / CRITICAL for this robot); change
them, re-run ``python3 -m maintenance_ai.train`` and the model follows.
"""

# --------------------------------------------------------------------------
# Data layout
# --------------------------------------------------------------------------
RAW_COLUMNS = [
    "linear_speed",
    "angular_speed",
    "motor_current",
    "motor_temperature",
    "battery_voltage",
    "battery_soc",
]
CSV_COLUMNS = ["timestamp"] + RAW_COLUMNS

LABELS = ["NORMAL", "WARNING", "CRITICAL"]          # index == severity

# --------------------------------------------------------------------------
# Sampling / windows (the CSV logger writes at 5 Hz)
# --------------------------------------------------------------------------
SAMPLE_HZ = 5.0
DT = 1.0 / SAMPLE_HZ
WINDOW = 25            # samples in a feature window  (~5 s)
MIN_WINDOW = 10        # samples needed before the first assessment (~2 s)
GAP_SECONDS = 1.0      # time gap that starts a new segment (matches logger freshness)
VOLTAGE_RESET_V = 0.5  # an upward voltage jump this large == new run (battery reset)
STALE_SECONDS = 1.0    # live inputs older than this are considered stale

# --------------------------------------------------------------------------
# Physical plausibility limits used by the cleaner (anything outside is a glitch)
# --------------------------------------------------------------------------
PLAUSIBLE = {
    "linear_speed": (0.0, 5.0),
    "angular_speed": (0.0, 10.0),
    "motor_current": (0.0, 30.0),
    "motor_temperature": (-20.0, 150.0),
    "battery_voltage": (0.0, 40.0),
    "battery_soc": (0.0, 100.0),
}

# --------------------------------------------------------------------------
# Constants mirrored from src/maintenance_monitor/src/maintenance_sensor_sim.cpp
# (used ONLY to generate synthetic training data and the demo fault overlay)
# --------------------------------------------------------------------------
SIM = {
    "idle_current": 0.5,
    "lin_gain": 4.0,
    "ang_gain": 2.5,
    "max_current": 15.0,
    "ambient": 28.0,
    "heat_coeff": 0.015,     # I^2 heating
    "cool_coeff": 0.035,     # cooling toward ambient
    "max_temp": 120.0,
    "v_full": 24.0,
    "v_empty": 18.0,
    "drop_idle": 0.00002,    # V per 0.1 s tick when idle
    "drop_per_amp": 0.00030, # V per 0.1 s tick per ampere when moving
    "tick": 0.1,
}

# --------------------------------------------------------------------------
# Health policy: (warning, critical) limits.  "hi" = value >= limit is bad,
# "lo" = value <= limit is bad.
# --------------------------------------------------------------------------
THRESHOLDS = {
    # absolute levels
    "overheating":         {"feature": "motor_temperature", "dir": "hi", "warn": 55.0,  "crit": 75.0},
    "overcurrent":         {"feature": "current_mean_5s",   "dir": "hi", "warn": 10.0,  "crit": 13.0},
    "low_battery":         {"feature": "battery_soc",       "dir": "lo", "warn": 30.0,  "crit": 15.0},
    # behaviour that the load does NOT explain (learned residuals)
    "abnormal_heating":    {"feature": "thermal_resid_cps", "dir": "hi", "warn": 0.20,  "crit": 0.50},
    "mechanical_drag":     {"feature": "cur_resid_5s",      "dir": "hi", "warn": 2.0,   "crit": 5.0},
    "battery_degradation": {"feature": "volt_resid_vps",    "dir": "lo", "warn": -0.010, "crit": -0.030},
}

# Order used to list reasons / pick the recommendation when severities tie:
# root causes first, symptoms last.
RULE_PRIORITY = ["mechanical_drag", "abnormal_heating", "battery_degradation",
                 "overcurrent", "overheating", "low_battery"]

MESSAGES = {
    "overheating": (
        "Motor temperature is {v:.1f} C",
        "Stop the robot and let the motor cool; inspect cooling and the load.",
    ),
    "abnormal_heating": (
        "Motor is heating {v:.2f} C/s faster than its load explains",
        "Check motor cooling/ventilation and winding or bearing condition.",
    ),
    "overcurrent": (
        "Average motor current is {v:.1f} A",
        "Reduce load or speed; check for a jammed wheel or drivetrain.",
    ),
    "mechanical_drag": (
        "Motor draws {v:.1f} A more than the motion requires",
        "Inspect wheels, bearings and drivetrain for friction or obstruction.",
    ),
    "low_battery": (
        "Battery state of charge is {v:.0f} %",
        "Return to the charger; schedule battery service if charge drops quickly.",
    ),
    "battery_degradation": (
        "Battery voltage is falling {v:.3f} V/s faster than the load explains",
        "Test battery health and connections; replace the pack if confirmed.",
    ),
}
ANOMALY_MESSAGE = (
    "Sensor pattern is unlike anything seen in normal operation",
    "Check sensor wiring/readings; compare current, speed and temperature for consistency.",
)

# --------------------------------------------------------------------------
# Anomaly detector behaviour
# --------------------------------------------------------------------------
ANOMALY_WINDOW = 15        # last N samples considered
ANOMALY_MIN_HITS = 8       # hits needed inside that window to raise an anomaly
ANOMALY_FALSE_ALARM = 0.01 # target per-sample false-alarm rate on normal data

# Human-friendly names used in explanations of anomaly / classifier decisions
FEATURE_LABELS = {
    "cur_resid_5s": "motor current vs. what the motion needs",
    "thermal_resid_cps": "motor heating vs. what the load explains",
    "volt_resid_vps": "battery voltage fall vs. what the load explains",
    "current_mean_5s": "average motor current",
    "current_std_5s": "motor current variability",
    "current_max_5s": "peak motor current",
    "temp_slope_cps": "motor temperature trend",
    "volt_slope_vps": "battery voltage trend",
    "temp_above_ambient": "motor temperature above ambient",
    "speed_mean_5s": "average linear speed",
    "ang_mean_5s": "average turning speed",
}
