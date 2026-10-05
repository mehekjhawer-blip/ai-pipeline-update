"""Step 7 - live AI inference from ROS 2 topics.

Subscribes : /odom, /motor_current, /motor_temperature, /battery_voltage, /battery_soc
             /maintenance/fault_inject   (std_msgs/String: overheat|drag|battery|stuck_current|clear)
Publishes  : /maintenance/health         (std_msgs/String, JSON of maintenance_ai.assessment.Assessment)

The node samples the latest values at 5 Hz (the rate the training data was
logged at).  If any input is older than 1 s it reports NO_DATA instead of
reusing stale values - the same "freshness-safe" rule as the CSV logger.

The existing C++ nodes are untouched; fault injection is an overlay applied
inside this node and is always reported in the ``injected_fault`` field.
"""

import math
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, String

from . import config as C
from .assessment import NO_DATA, Assessment
from .engine import HealthEngine
from .faults import KINDS, FaultInjector, FaultProfile
from .model import HealthModel
from .paths import model_path

FLOAT_TOPICS = {
    "/motor_current": "motor_current",
    "/motor_temperature": "motor_temperature",
    "/battery_voltage": "battery_voltage",
    "/battery_soc": "battery_soc",
}


class HealthNode(Node):
    def __init__(self) -> None:
        super().__init__("maintenance_ai")
        self.declare_parameter("model_path", "")
        self.declare_parameter("rate_hz", C.SAMPLE_HZ)
        path = self.get_parameter("model_path").value or str(model_path())
        self.model = HealthModel.load(Path(path))
        self.engine = HealthEngine(self.model)
        self.injector = FaultInjector()

        self._values, self._seen = {}, {}
        for topic, key in FLOAT_TOPICS.items():
            self.create_subscription(Float32, topic, lambda m, k=key: self._on_float(k, m.data), 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(String, "/maintenance/fault_inject", self._on_fault, 10)
        self.pub = self.create_publisher(String, "/maintenance/health", 10)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)
        self._last_status = None
        self.get_logger().info(f"AI health node started (model {path})")

    # -- inputs -----------------------------------------------------------
    def _touch(self, key: str, value: float) -> None:
        self._values[key] = value
        self._seen[key] = time.monotonic()

    def _on_float(self, key: str, value: float) -> None:
        self._touch(key, float(value))

    def _on_odom(self, msg: Odometry) -> None:
        t = msg.twist.twist
        self._touch("linear_speed", math.hypot(t.linear.x, t.linear.y))
        self._touch("angular_speed", abs(t.angular.z))

    def _on_fault(self, msg: String) -> None:
        cmd = msg.data.strip().lower()
        if cmd in KINDS:
            self.injector.set_fault(FaultProfile.demo(cmd))
            self.get_logger().warn(f"Demo fault injected: {cmd}")
        elif cmd in ("clear", "none", ""):
            self.injector.set_fault(None)
            self.get_logger().info("Demo fault cleared")
        else:
            self.get_logger().error(f"Unknown fault '{cmd}'. Use one of {KINDS} or 'clear'.")

    # -- main loop --------------------------------------------------------
    def _tick(self) -> None:
        now = time.monotonic()
        missing = [k for k in C.RAW_COLUMNS
                   if k not in self._seen or now - self._seen[k] > C.STALE_SECONDS]
        if missing:
            self.engine.reset()
            a = Assessment(status=NO_DATA, timestamp=time.time(),
                           reason="No fresh data from: " + ", ".join(missing),
                           injected_fault=self.injector.active)
        else:
            row = self.injector.apply({k: self._values[k] for k in C.RAW_COLUMNS})
            row["timestamp"] = time.time()
            a = self.engine.update(row, injected_fault=self.injector.active)
        self.pub.publish(String(data=a.to_json()))
        if a.status != self._last_status:
            self._last_status = a.status
            self.get_logger().info(f"Health: {a.status} - {a.reason}")


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = HealthNode()
    except Exception as exc:                      # e.g. model missing
        print(f"[maintenance_ai] cannot start: {exc}")
        rclpy.shutdown()
        raise SystemExit(1)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
