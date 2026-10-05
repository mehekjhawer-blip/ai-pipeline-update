"""Trained artefacts: classifier + anomaly detector + learned 'normal behaviour' models."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import ANOMALY_FEATURES, DERIVED, FEATURE_NAMES, ResidualModels, compute_features


@dataclass
class HealthModel:
    classifier: object                 # sklearn classifier with predict_proba
    iforest: object                    # sklearn IsolationForest (trained on healthy data only)
    anomaly_threshold: float           # score_samples below this == anomalous sample
    residuals: ResidualModels
    normal_mean: pd.Series
    normal_std: pd.Series
    meta: Dict = field(default_factory=dict)

    # -- inference helpers ---------------------------------------------------
    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_features(df, self.residuals)

    def proba(self, feats: pd.DataFrame) -> np.ndarray:
        """(n, 3) class probabilities in NORMAL/WARNING/CRITICAL order."""
        raw = self.classifier.predict_proba(feats[FEATURE_NAMES].to_numpy(float))
        full = np.zeros((len(feats), len(C.LABELS)))
        full[:, np.asarray(self.classifier.classes_, int)] = raw
        return full

    def anomaly_scores(self, feats: pd.DataFrame) -> np.ndarray:
        return self.iforest.score_samples(feats[ANOMALY_FEATURES].to_numpy(float))

    def z_scores(self, feats: pd.DataFrame, only=None) -> pd.DataFrame:
        cols = list(only or DERIVED)
        return (feats[cols] - self.normal_mean[cols]) / (self.normal_std[cols] + 1e-9)

    # -- persistence ---------------------------------------------------------
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "HealthModel":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model not found: {path}\nTrain it first:  python3 -m maintenance_ai.train")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(path)
        for w in caught:
            if "version" in str(w.message).lower():
                raise RuntimeError(
                    "The saved model was trained with a different scikit-learn version "
                    f"({w.message}). Re-train on this machine:  python3 -m maintenance_ai.train")
        return model
