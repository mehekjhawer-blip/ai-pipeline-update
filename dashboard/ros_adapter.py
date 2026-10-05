"""Small, project-specific ROS 2 subscriber for maintenance telemetry."""

import threading
import time
from collections import deque
from typing import Deque, Optional

try:
    from .telemetry import TelemetryReading, utc_now
except ImportError:  # Also support ``streamlit run dashboard/app.py``.
    from telemetry import TelemetryReading, utc_now

HEALTH_TOPIC = "/maintenance/health"
FAULT_TOPIC = "/maintenance/fault_inject"


class RosTelemetryBridge:
    """Spin subscriptions to the existing robot topics in a background thread."""

    TOPIC_KEYS = {
        "/motor_current": "motor_current",
        "/motor_temperature": "motor_temperature",
        "/battery_voltage": "battery_voltage",
        "/battery_soc": "battery_soc",
    }
    REQUIRED_KEYS = set(TOPIC_KEYS.values()) | {"linear_speed", "angular_speed"}

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values = {}
        self._received_at = {}
        self._history: Deque[TelemetryReading] = deque(maxlen=1500)
        self._health = None                       # latest AI assessment (dict)
        self._health_at = 0.0
        self._health_history: Deque[dict] = deque(maxlen=1500)
        self._fault_pub = None
        self._node = None
        self._executor = None
        self._thread = None
        self._started = False
        self._error: Optional[str] = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._started = True
            self._error = None
        self._thread = threading.Thread(target=self._spin, name="ros-telemetry", daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.executors import SingleThreadedExecutor
            from std_msgs.msg import Float32, String

            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = rclpy.create_node("maintenance_dashboard")
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)

            for topic, key in self.TOPIC_KEYS.items():
                self._node.create_subscription(
                    Float32,
                    topic,
                    lambda message, field=key: self._on_value(field, message.data),
                    10,
                )
            self._node.create_subscription(Odometry, "/odom", self._on_odom, 10)
            self._node.create_subscription(String, HEALTH_TOPIC, self._on_health, 10)
            self._fault_pub = self._node.create_publisher(String, FAULT_TOPIC, 10)
            self._string_msg = String
            while rclpy.ok():
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as exc:  # ROS 2 may be absent on the dashboard host.
            with self._lock:
                self._error = str(exc)

    def _on_value(self, key: str, value: float) -> None:
        self._record(key, float(value))

    def _on_odom(self, message) -> None:
        twist = message.twist.twist
        speed = (twist.linear.x ** 2 + twist.linear.y ** 2) ** 0.5
        angular_speed = abs(twist.angular.z)
        now = utc_now()
        with self._lock:
            self._values["linear_speed"] = speed
            self._values["angular_speed"] = angular_speed
            self._received_at["linear_speed"] = time.monotonic()
            self._received_at["angular_speed"] = time.monotonic()
            if self.REQUIRED_KEYS.issubset(self._values):
                self._history.append(TelemetryReading(timestamp=now, **self._values.copy()))

    def _on_health(self, message) -> None:
        """AI assessments published by the maintenance_ai health node (JSON)."""
        import json

        try:
            data = json.loads(message.data)
        except ValueError:
            return
        with self._lock:
            self._health = data
            self._health_at = time.monotonic()
            self._health_history.append(data)

    def health_snapshot(self):
        """Return ``(latest_assessment_dict_or_None, seconds_since_it_arrived, history_list)``."""
        with self._lock:
            age = time.monotonic() - self._health_at if self._health is not None else None
            return self._health, age, list(self._health_history)

    def send_fault(self, kind: str) -> bool:
        """Ask the AI node to overlay a demo fault ('clear' removes it)."""
        if self._fault_pub is None:
            return False
        self._fault_pub.publish(self._string_msg(data=kind))
        return True

    def _record(self, key: str, value: float) -> None:
        with self._lock:
            self._values[key] = value
            self._received_at[key] = time.monotonic()

    def snapshot(self):
        with self._lock:
            fresh = self.REQUIRED_KEYS.issubset(self._values) and all(
                time.monotonic() - self._received_at.get(key, 0) < 2.0
                for key in self.REQUIRED_KEYS
            )
            return fresh, list(self._history), self._error
