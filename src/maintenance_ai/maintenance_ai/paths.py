"""Locate the repository root (data/, models/, reports/) from source or install space."""

import os
from pathlib import Path
from typing import Optional

_MARKERS = ("data", "dashboard")


def _looks_like_root(path: Path) -> bool:
    return all((path / m).is_dir() for m in _MARKERS)


def find_repo_root(start: Optional[Path] = None) -> Path:
    candidates = []
    env = os.environ.get("MAINTENANCE_WS")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve())
    candidates.append(Path.home() / "robot_maintenance_ws")
    if start:
        candidates.insert(0, Path(start))
    for base in candidates:
        for parent in [base, *base.parents]:
            if _looks_like_root(parent):
                return parent
    return Path.cwd()


def model_path(root: Optional[Path] = None) -> Path:
    return (root or find_repo_root()) / "models" / "health_model.joblib"


def data_path(root: Optional[Path] = None) -> Path:
    return (root or find_repo_root()) / "data" / "maintenance_data.csv"
