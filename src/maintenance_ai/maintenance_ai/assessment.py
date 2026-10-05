"""Light-weight message type shared by the ROS node, the engine and the dashboard."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

WARMING_UP = "WARMING_UP"
NO_DATA = "NO_DATA"
STATUS_ORDER = ["NORMAL", "WARNING", "CRITICAL"]


@dataclass
class Assessment:
    status: str = WARMING_UP
    confidence: Optional[float] = None
    reason: str = "Collecting data for the first window"
    recommendation: str = ""
    source: str = ""                       # classifier | safety rule | anomaly detector
    anomaly: bool = False
    anomaly_score: Optional[float] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    injected_fault: str = "none"           # demo overlay active on the live stream
    timestamp: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(text: str) -> "Assessment":
        d = json.loads(text)
        return Assessment(**{k: d[k] for k in Assessment.__dataclass_fields__ if k in d})

    @property
    def timestamp_dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
