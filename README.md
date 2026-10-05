# AI-Based Robotic Maintenance Monitoring System

This project uses ROS 2 and Gazebo to simulate a robot and generate
maintenance telemetry for AI-based condition monitoring.

## Current Features

- ROS 2 Humble
- Gazebo Fortress / Ignition Gazebo
- Robot keyboard control
- ROS 2 `/cmd_vel`
- Gazebo `/odom`
- Motor current simulation
- Motor temperature simulation
- Battery voltage simulation
- Battery SOC simulation
- CSV dataset generation
- Live telemetry pipeline

## ROS 2 Packages

- `maintenance_robot`
- `maintenance_monitor`
- `maintenance_data_logger`

## Dataset

The generated dataset is stored at:

`data/maintenance_data.csv`

Columns:

- timestamp
- linear_speed
- angular_speed
- motor_current
- motor_temperature
- battery_voltage
- battery_soc

## Maintenance Dashboard

The dashboard reads the existing ROS 2 telemetry topics (`/odom`, `/motor_current`,
`/motor_temperature`, `/battery_voltage`, and `/battery_soc`). If ROS 2 is not
available, choose **Use Historical CSV** to view `data/maintenance_data.csv`.
Historical values are labeled as historical and are never presented as live.

The dashboard is a separate process and does not alter the Gazebo simulation,
ROS 2 publishers, or CSV logger. Run it from the repository root after sourcing
your ROS 2 installation and workspace:

```bash
python3 -m pip install -r dashboard/requirements.txt
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 -m streamlit run dashboard/app.py
```

In a separate terminal, start the existing robot and telemetry nodes as usual.
The dashboard subscribes to these project-specific ROS 2 topics directly.

To use historical data without ROS 2, install the dashboard requirements and
run `python3 -m streamlit run dashboard/app.py`; select **Use Historical CSV**
on the disconnected screen. A valid CSV must include all seven existing logger
columns: `timestamp`, `linear_speed`, `angular_speed`, `motor_current`,
`motor_temperature`, `battery_voltage`, and `battery_soc`.

The Maintenance Analysis page reports descriptive statistics and the current AI
condition assessment. The **AI Health** section shows the NORMAL / WARNING /
CRITICAL status, confidence, reason, recommended action, a status timeline and
demo fault-injection buttons (see below).

## AI Health Monitoring

A `maintenance_ai` ROS 2 package adds data cleaning, feature engineering,
NORMAL / WARNING / CRITICAL classification (Random Forest), anomaly detection
(Isolation Forest) and live inference. Assessments are published as JSON on
`/maintenance/health` and shown in the dashboard.

```bash
pip3 install --user -r requirements-ai.txt
colcon build --symlink-install && source install/setup.bash
python3 -m maintenance_ai.train          # ~3-4 min; writes models/health_model.joblib
python3 -m maintenance_ai.replay         # AI over data/maintenance_data.csv, no ROS needed
ros2 launch maintenance_ai full_demo.launch.py   # Gazebo + sensor sim + AI node
```

Full guide, demo script, results and limitations: [docs/AI_GUIDE.md](docs/AI_GUIDE.md).

