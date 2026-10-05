"""Fast checks for the AI pipeline.  Run:  python3 -m pytest src/maintenance_ai/test -q"""

import numpy as np
import pandas as pd
import pytest

from maintenance_ai import config as C
from maintenance_ai.assessment import Assessment
from maintenance_ai.engine import HealthEngine, assess_frame
from maintenance_ai.faults import FaultInjector, FaultProfile
from maintenance_ai.labeling import label_rows
from maintenance_ai.model import HealthModel
from maintenance_ai.paths import data_path, find_repo_root, model_path
from maintenance_ai.preprocess import clean
from maintenance_ai.simulate import simulate_session


@pytest.fixture(scope="module")
def model():
    p = model_path()
    if not p.is_file():
        pytest.skip("train first: python3 -m maintenance_ai.train")
    return HealthModel.load(p)


def _row(**kw):
    base = dict(timestamp=0.0, linear_speed=0.0, angular_speed=0.0, motor_current=0.5,
                motor_temperature=30.0, battery_voltage=24.0, battery_soc=100.0)
    base.update(kw)
    return base


# ---- preprocessing ---------------------------------------------------------
def test_clean_drops_bad_rows_and_segments():
    rows = [_row(timestamp=t * 0.2) for t in range(5)]
    rows += [_row(timestamp=100 + t * 0.2) for t in range(5)]          # time gap
    rows.append(_row(timestamp=0.2))                                    # duplicate timestamp
    rows.append(_row(timestamp=50.0, motor_current=float("nan")))      # NaN
    rows.append(_row(timestamp=60.0, battery_soc=250.0))                # impossible SOC
    df, rep = clean(pd.DataFrame(rows))
    assert rep.rows_out == 10 and rep.segments == 2
    assert rep.dropped["duplicate_timestamp"] == 1


def test_clean_requires_columns():
    with pytest.raises(ValueError):
        clean(pd.DataFrame({"timestamp": [1.0]}))


# ---- labels ------------------------------------------------------------------
def test_policy_labels():
    f = pd.DataFrame([
        dict(motor_temperature=30, current_mean_5s=2, battery_soc=90, thermal_resid_cps=0, cur_resid_5s=0, volt_resid_vps=0),
        dict(motor_temperature=60, current_mean_5s=2, battery_soc=90, thermal_resid_cps=0, cur_resid_5s=0, volt_resid_vps=0),
        dict(motor_temperature=30, current_mean_5s=2, battery_soc=10, thermal_resid_cps=0, cur_resid_5s=0, volt_resid_vps=0),
        dict(motor_temperature=30, current_mean_5s=2, battery_soc=90, thermal_resid_cps=0, cur_resid_5s=2.5, volt_resid_vps=0),
    ])
    assert label_rows(f).tolist() == [0, 1, 2, 1]


# ---- faults ------------------------------------------------------------------
@pytest.mark.parametrize("kind,key,sign", [("overheat", "motor_temperature", 1),
                                           ("drag", "motor_current", 1),
                                           ("battery", "battery_voltage", -1)])
def test_fault_moves_signal(kind, key, sign):
    inj = FaultInjector()
    inj.set_fault(FaultProfile.demo(kind))
    out = None
    for _ in range(200):
        out = inj.apply(_row())
    assert sign * (out[key] - _row()[key]) > 0


def test_no_fault_is_identity():
    assert FaultInjector().apply(_row()) == _row()


# ---- assessment message -------------------------------------------------------
def test_assessment_json_roundtrip():
    a = Assessment(status="WARNING", confidence=0.9, reason="x", timestamp=12.0,
                   probabilities={"NORMAL": 0.1, "WARNING": 0.9, "CRITICAL": 0.0})
    assert Assessment.from_json(a.to_json()) == a


# ---- model-dependent ---------------------------------------------------------
def test_real_csv_is_healthy(model):
    from maintenance_ai.preprocess import load_and_clean
    path = data_path(find_repo_root())
    if not path.is_file():
        pytest.skip("no data/maintenance_data.csv")
    df, _ = load_and_clean(path)
    res = assess_frame(model, df, explain_rows=False)
    res = res[res.status_idx >= 0]
    assert (res.status_idx == 0).mean() > 0.97


@pytest.mark.parametrize("kind", ["overheat", "drag", "battery"])
def test_faults_are_detected_and_escalate(model, kind):
    df = simulate_session(np.random.default_rng(5), 200, FaultProfile.demo(kind),
                          fault_onset_s=30)[C.CSV_COLUMNS]
    res = assess_frame(model, df.assign(segment=0), explain_rows=False)
    after = res.iloc[int(35 / C.DT):]
    assert (after.status_idx > 0).mean() > 0.9
    assert res.status_idx.iloc[:int(25 / C.DT)].max() == 0       # quiet before onset


def test_streaming_matches_batch(model):
    df = simulate_session(np.random.default_rng(11), 60, FaultProfile.demo("drag"),
                          fault_onset_s=20)[C.CSV_COLUMNS]
    batch = assess_frame(model, df.assign(segment=0), explain_rows=False)["status"].tolist()
    eng = HealthEngine(model)
    live = [eng.update(r).status for r in df.to_dict("records")]
    assert live == batch


def test_engine_resets_after_gap(model):
    eng = HealthEngine(model)
    for k in range(30):
        eng.update(_row(timestamp=k * 0.2))
    a = eng.update(_row(timestamp=500.0))        # long gap -> new segment
    assert a.status == "WARMING_UP"


def test_inference_is_fast_enough_for_5hz(model):
    import time
    eng = HealthEngine(model)
    rows = simulate_session(np.random.default_rng(2), 30)[C.CSV_COLUMNS].to_dict("records")
    t = time.perf_counter()
    for r in rows:
        eng.update(r)
    per_update = (time.perf_counter() - t) / len(rows)
    assert per_update < 0.1                       # budget is 0.2 s per sample
