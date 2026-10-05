# AI Health Monitoring - Guide

This covers everything that was "still left" after the telemetry/logging work.

| Remaining item | Where it lives |
|---|---|
| Data cleaning / preprocessing | `src/maintenance_ai/maintenance_ai/preprocess.py` |
| Feature engineering | `features.py` (rolling stats + 3 learned "normal behaviour" residuals) |
| NORMAL / WARNING / CRITICAL labels | `labeling.py`, thresholds in `config.py` |
| AI model training | `train.py` (Random Forest classifier) |
| Anomaly detection | `train.py` + `engine.py` (Isolation Forest on the residuals) |
| Save / export model | `models/health_model.joblib`, `models/model_card.json`, `reports/*.png` |
| Live AI inference from ROS 2 topics | `ros_node.py` -> node `health_node`, topic `/maintenance/health` |
| Live dashboard + AI output connected | `dashboard/app.py` -> **AI Health** section |
| Final integrated demo | `launch/full_demo.launch.py` + the demo script below |

## Quick start

```bash
cd ~/robot_maintenance_ws
pip3 install --user -r requirements-ai.txt      # numpy<2 is deliberate (ROS 2 Humble)
colcon build --symlink-install
source /opt/ros/humble/setup.bash && source install/setup.bash

python3 -m maintenance_ai.train                 # ~3-4 min, retrains on this machine
python3 -m pytest src/maintenance_ai/test -q    # 15 checks
python3 -m maintenance_ai.replay                # no ROS: AI over your logger CSV
python3 -m maintenance_ai.replay --simulate 150 --fault overheat --onset 30
```

Without `colcon build` you can prefix commands with `PYTHONPATH=src/maintenance_ai`.

## How it works

```
/odom + 4 sensor topics --(5 Hz sampler, freshness check)--> health_node
   |  [demo only] fault overlay (/maintenance/fault_inject)
   v
sliding 5 s window -> features -> Random Forest  -> P(NORMAL/WARNING/CRITICAL)
                              \-> Isolation Forest -> "unlike healthy data?"
                               \-> policy rules   -> safety net + human reasons
   final = max(classifier, rules); NORMAL + persistent anomaly -> WARNING
   -> /maintenance/health (JSON: status, confidence, reason, recommendation, ...)
   -> dashboard (live)       |  same code on the CSV -> dashboard (historical)
```

**Residual features** are the key idea. Three small linear models are learned from
healthy data and compare *what the robot does* with *what its load explains*:
motor current vs. speed, heating rate vs. I^2 and cooling, voltage fall vs. current.
A drifting temperature at idle or a current that does not match the motion shows up
immediately, regardless of driving style. They recovered your simulator constants
almost exactly (heating 0.015, cooling 0.035, current gains 4.0 and 2.5).

**Safety net.** The rules in `config.THRESHOLDS` always apply, so a CRITICAL value
can never be hidden by a model mistake. The classifier adds probabilities and early
warnings; the anomaly detector catches faults nobody wrote a rule for.

**One code path.** Training, CSV replay and live inference all use the same
`compute_features` / `assess_frame`; a test checks that streaming equals batch.

## Final demo script (about 5 minutes)

Terminals (after build + train):

```bash
ros2 launch maintenance_ai full_demo.launch.py        # 1: Gazebo, sensor sim, AI node
ros2 run maintenance_robot keyboard_control           # 2: drive the robot
python3 -m streamlit run dashboard/app.py             # 3: dashboard -> AI Health
```

1. Drive for a minute: status stays NORMAL, residual-based reasons stay quiet.
2. In **AI Health** click a fault button. Typical behaviour (simulated, see replay):

   | Fault | First WARNING | CRITICAL |
   |---|---|---|
   | Overheating | ~2-3 s | ~25 s |
   | Mechanical drag | ~5 s | ~45-50 s |
   | Battery degradation | ~3-6 s | ~45 s |
   | Stuck current sensor (never seen in training) | ~3-8 s via anomaly detector | ~30-37 s |

3. Show the reason + recommended action, the probabilities and the timeline.
4. Click **Clear fault**. Everything the dashboard shows during a demo fault is
   labelled *SIMULATED FAULT ACTIVE*; the C++ nodes are never modified.
5. Natural fault without buttons: keep driving ~5-6 minutes; the accelerated battery
   model drops SOC below 30 %, which raises a real low-battery WARNING.

Without ROS: `python3 -m streamlit run dashboard/app.py` -> **Use Historical CSV** ->
**AI Health** runs the same model over `data/maintenance_data.csv`.

## What the numbers mean (read before presenting)

Results from the last training run (`models/model_card.json`):

- Your real CSV (484 assessed rows): 100 % NORMAL, 0 anomaly flags.
- Unseen healthy synthetic sessions: 0.3 % false alarms.
- Faults flagged after onset: overheat 98.7 %, drag 98.1 %, battery 97.4 %,
  stuck sensor (unseen) 94.5 % - all in a few seconds.
- Classifier cross-validated macro-F1 ~ 0.999.

Be honest about what they show:

- **The F1 is high by construction.** Labels come from your thresholds, so it
  measures how well the model reproduces that policy, not real-world fault
  detection. Its value is probabilities, early warnings and a deployable artefact.
- **Training faults are simulated.** The real CSV has only ~100 s of healthy
  driving (no WARNING/CRITICAL). Synthetic sessions replay your simulator's own
  equations with random driving, then inject faults (`faults.py`).
- **Healthy-looking real data is the real test**, and it passes. A real fault on
  hardware would need real examples.

## Tuning, retraining, extending

- **Change what counts as WARNING/CRITICAL:** edit `THRESHOLDS` in `config.py`, then retrain.
- **More real data:** the logger appends to the CSV; each run is detected as a
  separate segment. More real runs improve the learned residual models.
- **Better demo faults:** `FaultProfile.demo()` in `faults.py` sets growth speed.
- **Model/scikit-learn mismatch:** the loader tells you to retrain; just run `train` again.

## Troubleshooting

- *"Model not found"*: run `python3 -m maintenance_ai.train` from the repo root.
- *Dashboard says no AI assessments*: start `health_node` (the full-demo launch does).
- *`NO_DATA` status*: a topic is older than 1 s (sensor sim or Gazebo bridge not running).
- *numpy / cv_bridge errors after pip install*: you installed NumPy 2; use `requirements-ai.txt`.

## Verified vs not verified

Verified here: cleaning, features, training, evaluation, streaming == batch,
15 unit tests, replay CLI, and the dashboard (historical mode and a live-mode
simulation using a stand-in for the ROS bridge).
**Not run here:** Gazebo, ROS 2 topics, `colcon build` and the launch file - there
was no ROS 2 installation. Expect to fix small integration details on first run.
