"""Maintenance robot dashboard. Run with: streamlit run dashboard/app.py"""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

try:
    from .analysis import METRICS, statistics
    from .csv_adapter import CsvDataError, load_csv
    from .ros_adapter import RosTelemetryBridge
except ImportError:  # Streamlit executes this file as a top-level script.
    from analysis import METRICS, statistics
    from csv_adapter import CsvDataError, load_csv
    from ros_adapter import RosTelemetryBridge


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "maintenance_data.csv"
MODEL_PATH = ROOT / "models" / "health_model.joblib"

# The AI package lives in src/maintenance_ai (also importable after ``colcon build``).
_AI_SRC = ROOT / "src" / "maintenance_ai"
if _AI_SRC.is_dir() and str(_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_AI_SRC))
try:
    from maintenance_ai import config as AI_CFG
    from maintenance_ai.assessment import NO_DATA, WARMING_UP
    from maintenance_ai.engine import assess_frame
    from maintenance_ai.model import HealthModel
    from maintenance_ai.preprocess import clean as ai_clean
    AI_IMPORT_ERROR = None
except Exception as exc:  # scikit-learn / package missing: dashboard still works without AI
    AI_IMPORT_ERROR = str(exc)

STATUS_COLORS = {"NORMAL": "#2FBF71", "WARNING": "#F2B705", "CRITICAL": "#E5484D",
                 "WARMING_UP": "#7B8794", "NO_DATA": "#7B8794"}
FAULT_BUTTONS = [("Overheating", "overheat"), ("Mechanical drag", "drag"),
                 ("Battery degradation", "battery"), ("Stuck current sensor", "stuck_current"),
                 ("Clear fault", "clear")]
st.set_page_config(page_title="Robot Maintenance Monitor", page_icon="⚙", layout="wide")


@st.cache_resource
def get_ros_bridge():
    bridge = RosTelemetryBridge()
    bridge.start()
    return bridge


def readings_frame(readings):
    return pd.DataFrame([reading.__dict__ for reading in readings]).sort_values("timestamp")


def render_chart(frame, title, fields, labels, time_window):
    filtered = frame
    if time_window != "All available":
        duration = {"5 minutes": 5, "15 minutes": 15, "1 hour": 60}[time_window]
        filtered = frame[frame.timestamp >= frame.timestamp.max() - timedelta(minutes=duration)]
    fig = px.line(filtered, x="timestamp", y=fields, labels=labels, title=title)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=48, b=12), legend_title_text="",
    )
    st.plotly_chart(fig, width="stretch")


def metric_card(label, value, unit, status=None):
    suffix = f" {unit}" if unit else ""
    st.metric(label, f"{value:.2f}{suffix}" if value is not None else "—", delta=status)


@st.cache_resource
def get_ai_model():
    """Load the trained model once; returns ``(model_or_None, error_or_None)``."""
    if AI_IMPORT_ERROR:
        return None, f"AI package could not be imported: {AI_IMPORT_ERROR}"
    try:
        return HealthModel.load(MODEL_PATH), None
    except Exception as exc:
        return None, str(exc)


def historical_assessments(readings):
    """Run the model over the loaded CSV (cached per loaded file)."""
    if st.session_state.get("ai_hist_for") is not readings:
        model, error = get_ai_model()
        if model is None:
            st.session_state.ai_hist = (None, error)
        else:
            raw = pd.DataFrame([
                {"timestamp": r.timestamp.timestamp(), "linear_speed": r.linear_speed,
                 "angular_speed": r.angular_speed, "motor_current": r.motor_current,
                 "motor_temperature": r.motor_temperature, "battery_voltage": r.battery_voltage,
                 "battery_soc": r.battery_soc} for r in readings])
            cleaned, _ = ai_clean(raw)
            result = assess_frame(model, cleaned)
            result.insert(0, "timestamp", pd.to_datetime(cleaned["timestamp"], unit="s", utc=True))
            st.session_state.ai_hist = (result, None)
        st.session_state.ai_hist_for = readings
    return st.session_state.ai_hist


def assessment_from_row(row):
    probs = {lab: float(row[f"p_{lab.lower()}"]) for lab in AI_CFG.LABELS
             if row[f"p_{lab.lower()}"] == row[f"p_{lab.lower()}"]}
    return {"status": row["status"], "confidence": None if row["confidence"] != row["confidence"] else float(row["confidence"]),
            "reason": row["reason"], "recommendation": row["recommendation"], "source": row["source"],
            "anomaly": bool(row["anomaly"]), "anomaly_score": None if row["anomaly_score"] != row["anomaly_score"] else float(row["anomaly_score"]),
            "probabilities": probs, "injected_fault": "none", "timestamp": row["timestamp"].timestamp()}


def current_assessment(bridge, historical, readings):
    """Return ``(assessment_dict_or_None, history_dataframe, message_or_None)``."""
    if historical:
        result, error = historical_assessments(readings)
        if result is None:
            return None, None, error
        valid = result[result.status_idx >= 0]
        return (assessment_from_row(result.iloc[-1]), valid, None)
    latest, age, history = bridge.health_snapshot()
    if latest is None or (age is not None and age > 3.0):
        return None, None, ("No AI assessments are arriving. Start the AI node:  "
                            "ros2 run maintenance_ai health_node")
    frame = pd.DataFrame(history)
    frame = frame[frame.status.isin(AI_CFG.LABELS)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    frame["status_idx"] = frame.status.map({lab: i for i, lab in enumerate(AI_CFG.LABELS)})
    return latest, frame, None


def render_assessment_card(a, compact=False):
    color = STATUS_COLORS.get(a["status"], "#7B8794")
    conf = f"{a['confidence'] * 100:.0f}% confidence" if a.get("confidence") is not None else ""
    source = f" &middot; via {a['source']}" if a.get("source") else ""
    rec = ("" if compact or not a.get("recommendation") else
           f"<div style='margin-top:.35rem;color:#C9D1D9'><b>Recommended action:</b> {a['recommendation']}</div>")
    st.markdown(
        f"""<div style="border-left:6px solid {color};background:rgba(255,255,255,0.04);
        padding:.8rem 1rem;border-radius:6px;margin:.2rem 0 1rem">
        <span style="background:{color};color:#0B0F14;font-weight:700;padding:.1rem .6rem;
        border-radius:999px;letter-spacing:.04em">{a['status'].replace('_', ' ')}</span>
        <span style="color:#98A4AF;margin-left:.6rem">{conf}{source}</span>
        <div style="margin-top:.45rem;font-size:1.02rem">{a['reason']}</div>{rec}</div>""",
        unsafe_allow_html=True)


def status_timeline(frame):
    fig = px.scatter(frame, x="timestamp", y="status_idx", color="status",
                     color_discrete_map=STATUS_COLORS, category_orders={"status": AI_CFG.LABELS})
    fig.update_traces(marker=dict(size=5))
    fig.update_yaxes(tickvals=[0, 1, 2], ticktext=AI_CFG.LABELS, range=[-0.4, 2.4], title=None)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=12, r=12, t=40, b=12), legend_title_text="", title="Health status over time")
    st.plotly_chart(fig, width="stretch")


def render_ai_health(bridge, historical, readings):
    st.markdown("### AI health assessment")
    if AI_IMPORT_ERROR:
        st.error(f"AI package unavailable: {AI_IMPORT_ERROR}")
        return
    assessment, frame, message = current_assessment(bridge, historical, readings)
    if assessment is None:
        st.info(message or "No assessment available yet.")
        if message and "model" in message.lower() and "train" in message.lower():
            st.code("python3 -m maintenance_ai.train", language="bash")
    else:
        if historical:
            st.caption("Historical assessment of the loaded CSV - not live. Showing the last recorded sample.")
        if assessment.get("injected_fault", "none") != "none":
            st.warning(f"SIMULATED FAULT ACTIVE: {assessment['injected_fault']} (demo overlay, not a real robot fault)")
        render_assessment_card(assessment)
        left, mid, right = st.columns(3)
        left.metric("Anomaly detector", "ANOMALOUS" if assessment["anomaly"] else "Normal")
        score = assessment.get("anomaly_score")
        mid.metric("Anomaly score", f"{score:.3f}" if score is not None else "-",
                   help="Lower is stranger. The alarm threshold is learned from healthy data.")
        probs = assessment.get("probabilities") or {}
        right.metric("Classifier P(critical)", f"{probs.get('CRITICAL', 0.0) * 100:.0f}%" if probs else "-")
        if probs:
            bars = pd.DataFrame({"class": list(probs), "probability": list(probs.values())})
            fig = px.bar(bars, x="probability", y="class", orientation="h", color="class",
                         color_discrete_map=STATUS_COLORS, range_x=[0, 1], title="Classifier probabilities")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=12, r=12, t=40, b=12), showlegend=False, height=200)
            st.plotly_chart(fig, width="stretch")
        if frame is not None and len(frame):
            status_timeline(frame)
            counts = frame["status"].value_counts()
            st.caption("Samples by status: " + " | ".join(f"{k} {v:,}" for k, v in counts.items()))

    if not historical:
        st.markdown("#### Demo: inject a simulated fault")
        st.caption("Overlays a fault on the live telemetry inside the AI node so you can show the alerts "
                   "end to end. The robot and C++ nodes are not changed; assessments are labelled as simulated.")
        columns = st.columns(len(FAULT_BUTTONS))
        for column, (label, kind) in zip(columns, FAULT_BUTTONS):
            if column.button(label, key=f"fault_{kind}", width="stretch"):
                if bridge.send_fault(kind):
                    st.toast(f"Sent: {label}")
                else:
                    st.error("Not connected to ROS 2, cannot send the command.")


def main():
    st.markdown(
        """
        <style>
        .dashboard-heading {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            margin: 0 0 0.2rem;
        }
        .dashboard-heading__accent {
            width: 4px;
            height: 2.6rem;
            flex: 0 0 4px;
            border-radius: 4px;
            background: #68DDB3;
        }
        .dashboard-heading h1 {
            margin: 0;
            color: #F2F5F7;
            font-family: Inter, "Segoe UI", Roboto, system-ui, sans-serif;
            font-size: clamp(2.15rem, 3.6vw, 2.75rem);
            font-weight: 700;
            letter-spacing: -0.045em;
            line-height: 1.12;
        }
        .dashboard-subtitle {
            margin: 0.35rem 0 1.5rem;
            color: #98A4AF;
            font-size: 0.98rem;
            letter-spacing: 0.01em;
        }
        @media (max-width: 600px) {
            .dashboard-heading h1 { font-size: 2rem; }
            .dashboard-heading__accent { height: 2.25rem; }
        }
        </style>
        <div class="dashboard-heading">
            <span class="dashboard-heading__accent" aria-hidden="true"></span>
            <h1>Robot Maintenance Monitor</h1>
        </div>
        <p class="dashboard-subtitle">ROS 2 telemetry for the simulated maintenance robot</p>
        """,
        unsafe_allow_html=True,
    )
    bridge = get_ros_bridge()
    connected, live_readings, bridge_error = bridge.snapshot()

    if "source" not in st.session_state:
        st.session_state.source = "live" if connected else "disconnected"

    if st.session_state.source == "live" and not connected:
        st.session_state.source = "disconnected"

    if st.session_state.source == "disconnected":
        st_autorefresh(interval=2000, key="connection_check")
        if connected:
            st.session_state.source = "live"
            st.rerun()
        st.error("ROS 2 Disconnected" + (f" · {bridge_error}" if bridge_error else ""))
        left, right = st.columns(2)
        if left.button("Retry Connection", type="primary", width="stretch"):
            bridge.start()
            st.session_state.source = "retrying"
            st.session_state.retry_started = time.time()
            st.rerun()
        if right.button("Use Historical CSV", width="stretch"):
            try:
                st.session_state.csv_readings = load_csv(CSV_PATH)
                st.session_state.source = "historical"
                st.rerun()
            except CsvDataError as exc:
                st.error(str(exc))
        return

    if st.session_state.source == "retrying":
        if connected:
            st.session_state.source = "live"
            st.rerun()
        if time.time() - st.session_state.get("retry_started", 0) > 6:
            st.session_state.source = "disconnected"
            st.rerun()
        st.info("Attempting to connect to ROS 2 telemetry…")
        left, right = st.columns(2)
        if left.button("Retry Connection", type="primary", width="stretch"):
            bridge.start()
            st.session_state.retry_started = time.time()
            st.rerun()
        if right.button("Use Historical CSV", width="stretch"):
            try:
                st.session_state.csv_readings = load_csv(CSV_PATH)
                st.session_state.source = "historical"
                st.rerun()
            except CsvDataError as exc:
                st.error(str(exc))
        st_autorefresh(interval=2000, key="retry_refresh")
        return

    historical = st.session_state.source == "historical"
    if historical:
        readings = st.session_state.csv_readings
        st.warning("Historical Data · Loaded from data/maintenance_data.csv")
    else:
        readings = live_readings
        st.success("ROS 2 Connected · Live telemetry")
        st.caption(f"Last update: {readings[-1].timestamp.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}" if readings else "Waiting for the first complete telemetry sample…")

    if not readings:
        st.info("Waiting for readings from /odom and the maintenance sensor topics.")
        st.button("Retry Connection", on_click=lambda: st.rerun())
        st.button("Use Historical CSV", on_click=lambda: st.session_state.update(source="historical"))
        return

    frame = readings_frame(readings)
    nav = st.radio("Section", ["Overview", "AI Health", "Sensor Trends", "Maintenance Analysis"], horizontal=True, label_visibility="collapsed")
    if nav == "Overview":
        latest = readings[-1]
        ai_assessment, _, _ = current_assessment(bridge, historical, readings) if AI_IMPORT_ERROR is None else (None, None, None)
        if ai_assessment is not None:
            render_assessment_card(ai_assessment, compact=True)
        st.markdown("### Current telemetry")
        metric_columns = st.columns(6)
        metrics = [
            ("Battery voltage", latest.battery_voltage, "V"),
            ("Battery SOC", latest.battery_soc, "%"),
            ("Motor current", latest.motor_current, "A"),
            ("Motor temperature", latest.motor_temperature, "°C"),
            ("Linear speed", latest.linear_speed, "m/s"),
            ("Angular speed", latest.angular_speed, "rad/s"),
        ]
        for column, (label, value, unit) in zip(metric_columns, metrics):
            with column:
                metric_card(label, value, unit)
        time_window = st.selectbox("Chart range", ["5 minutes", "15 minutes", "1 hour", "All available"], index=3, key="overview_range")
        left, right = st.columns(2)
        with left:
            render_chart(frame, "Motor temperature", ["motor_temperature"], {"motor_temperature": "°C"}, time_window)
            render_chart(frame, "Battery voltage", ["battery_voltage"], {"battery_voltage": "V"}, time_window)
        with right:
            render_chart(frame, "Motor current", ["motor_current"], {"motor_current": "A"}, time_window)
            render_chart(frame, "Robot motion", ["linear_speed", "angular_speed"], {"linear_speed": "Linear speed (m/s)", "angular_speed": "Angular speed (rad/s)"}, time_window)
        render_chart(frame, "Battery state of charge", ["battery_soc"], {"battery_soc": "%"}, time_window)
    elif nav == "AI Health":
        render_ai_health(bridge, historical, readings)
    elif nav == "Sensor Trends":
        st.markdown("### Sensor trends")
        time_window = st.selectbox("Time range", ["5 minutes", "15 minutes", "1 hour", "All available"], index=3)
        metric = st.selectbox("Parameter", ["Motor temperature", "Motor current", "Battery voltage", "Battery SOC", "Linear speed", "Angular speed"])
        field, unit = {
            "Motor temperature": ("motor_temperature", "°C"), "Motor current": ("motor_current", "A"),
            "Battery voltage": ("battery_voltage", "V"), "Battery SOC": ("battery_soc", "%"),
            "Linear speed": ("linear_speed", "m/s"), "Angular speed": ("angular_speed", "rad/s"),
        }[metric]
        render_chart(frame, metric, [field], {field: unit}, time_window)
        st.caption(f"{len(frame):,} telemetry samples")
    else:
        st.markdown("### Descriptive maintenance statistics")
        st.caption("Statistics summarize the currently selected data source and available time span.")
        for label, (field, unit) in METRICS.items():
            stats = statistics(readings, field)
            st.markdown(f"#### {label}")
            columns = st.columns(4)
            for column, (name, value) in zip(columns, stats.items() if stats else []):
                with column:
                    metric_card(name, value, unit)
        st.divider()
        st.markdown("#### AI condition assessment")
        ai_assessment, _, ai_message = current_assessment(bridge, historical, readings) if AI_IMPORT_ERROR is None else (None, None, AI_IMPORT_ERROR)
        if ai_assessment is not None:
            render_assessment_card(ai_assessment)
            st.caption("Open the AI Health section for probabilities, the status timeline and demo fault injection.")
        else:
            st.info(ai_message or "No condition assessment is available yet.")

    if not historical:
        st.caption("Live view refreshes every 2 seconds.")
        st_autorefresh(interval=2000, key="live_refresh")
    else:
        if st.button("Return to ROS 2 Live", width="content"):
            st.session_state.source = "live" if connected else "disconnected"
            st.rerun()


main()
