"""Steps 4-6 - train the models, evaluate them honestly, export them.

    python3 -m maintenance_ai.train            # from the repo root

Pipeline: clean real CSV -> synthetic sessions (normal + faults) -> learn
'normal behaviour' residual models -> features -> policy labels ->
RandomForest classifier + IsolationForest anomaly detector -> cross-validated
metrics -> ``models/health_model.joblib`` + ``models/model_card.json`` + plots.
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from . import __version__
from . import config as C
from .engine import assess_frame
from .features import ANOMALY_FEATURES, FEATURE_NAMES, compute_features, fit_residual_models
from .labeling import label_rows, rule_severities
from .model import HealthModel
from .paths import data_path, find_repo_root
from .preprocess import load_and_clean
from .simulate import build_dataset

STRIDE = 2   # neighbouring 5 Hz rows are near-duplicates; thin them for speed


def _rf(seed):
    return RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=3,
                                  class_weight="balanced", n_jobs=-1, random_state=seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=None, help="real logger CSV")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="fewer sessions (fast smoke run)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    root = find_repo_root()
    t_start = time.time()
    print(f"[1/7] Cleaning real data ({args.data or data_path(root)})")
    real, rep = load_and_clean(args.data or data_path(root))
    print("      ", rep)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    real.to_csv(root / "data" / "processed" / "maintenance_clean.csv", index=False)

    print("[2/7] Generating synthetic sessions (normal + injected faults)")
    k = 0.4 if args.quick else 1.0
    syn = build_dataset(seed=args.seed, n_normal=int(70 * k), n_per_fault=int(30 * k),
                        n_stuck=int(20 * k))
    syn["segment"] = syn["session"]
    print("       sessions:", syn.groupby("fault_kind")["session"].nunique().to_dict(),
          "| rows:", len(syn))

    print("[3/7] Learning 'normal behaviour' models and engineering features")
    healthy = syn[syn.fault_kind == "none"]
    fit_df = pd.concat([real.assign(segment=-1), healthy[real.columns]], ignore_index=True)
    residuals = fit_residual_models(fit_df)
    feats = compute_features(syn[C.CSV_COLUMNS + ["segment"]], residuals)
    labels = label_rows(feats)
    ok = feats.notna().all(axis=1)
    syn_ok = syn[ok].copy()
    X, y = feats[ok], labels[ok].astype(int)
    print("       label counts:", {C.LABELS[i]: int((y == i).sum()) for i in range(3)})
    sev = rule_severities(X)
    nh = syn_ok.fault_kind == "none"
    print("       non-NORMAL rows in fault-free sessions (should be low-battery only):",
          {c: int((sev[c][nh] > 0).sum()) for c in sev if (sev[c][nh] > 0).any()})

    train_kinds = syn_ok.fault_kind != "stuck_current"   # keep this fault UNSEEN
    sessions = syn_ok["session"].to_numpy()

    print("[4/7] Cross-validating the classifier (grouped by session)")
    idx = np.flatnonzero(train_kinds.to_numpy())
    pred = np.zeros(len(X), int)
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf.split(idx, groups=sessions[idx])):
        tr, te = idx[tr][::STRIDE], idx[te]
        m = _rf(args.seed).fit(X.iloc[tr].to_numpy(), y.iloc[tr].to_numpy())
        pred[te] = m.predict(X.iloc[te].to_numpy())
    cv_f1 = f1_score(y.iloc[idx], pred[idx], average="macro")
    cm = confusion_matrix(y.iloc[idx], pred[idx], labels=[0, 1, 2])
    print(f"       macro-F1 = {cv_f1:.3f}")
    print(classification_report(y.iloc[idx], pred[idx], target_names=C.LABELS, digits=3))

    print("[5/7] Training final classifier + anomaly detector")
    clf = _rf(args.seed).fit(X.iloc[idx[::STRIDE]].to_numpy(), y.iloc[idx[::STRIDE]].to_numpy())

    healthy_mask = (nh & (y == 0)).to_numpy()
    h_sessions = np.unique(sessions[healthy_mask])
    rng = np.random.default_rng(args.seed)
    val_s = set(rng.choice(h_sessions, size=max(2, len(h_sessions) // 5), replace=False))
    in_val = np.isin(sessions, list(val_s))
    fit_rows = np.flatnonzero(healthy_mask & ~in_val)[::STRIDE]
    iso = IsolationForest(n_estimators=200, max_samples=4096, random_state=args.seed, n_jobs=-1)
    iso.fit(X.iloc[fit_rows][ANOMALY_FEATURES].to_numpy())
    val_scores = iso.score_samples(X[healthy_mask & in_val][ANOMALY_FEATURES].to_numpy())
    threshold = float(np.quantile(val_scores, C.ANOMALY_FALSE_ALARM))

    normal_X = X[healthy_mask]
    model = HealthModel(
        classifier=clf, iforest=iso, anomaly_threshold=threshold, residuals=residuals,
        normal_mean=normal_X.mean(), normal_std=normal_X.std(),
        meta={"version": __version__,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "sklearn": sklearn.__version__, "python": platform.python_version(),
              "features": FEATURE_NAMES, "thresholds": C.THRESHOLDS, "seed": args.seed})

    print("[6/7] Evaluating end-to-end (final status incl. safety rules + persistence)")
    metrics = {"classifier_cv_macro_f1": round(float(cv_f1), 4),
               "classifier_confusion_matrix[NORMAL,WARNING,CRITICAL]": cm.tolist()}

    # (a) held-out healthy sessions -> false alarms ; (b) fault sessions -> detection
    rows = []
    for sid, g in syn.groupby("session"):
        if g.fault_kind.iloc[0] == "none" and sid not in val_s:
            continue                                  # only evaluate on unseen healthy sessions
        a = assess_frame(model, g[C.CSV_COLUMNS + ["segment"]], explain_rows=False)
        a["fault_kind"], a["fault_active"], a["session"] = g.fault_kind.values, g.fault_active.values, sid
        a["label"] = labels.reindex(g.index).values
        rows.append(a)
    ev = pd.concat(rows)
    ev = ev[ev.status_idx >= 0]
    h = ev[ev.fault_kind == "none"]
    metrics["healthy_sessions_unseen"] = {
        "rows": int(len(h)),
        "false_alarm_rate": round(float(((h.status_idx > 0) & (h.label == 0)).mean()), 4),
        "missed_policy_events": round(float(((h.status_idx == 0) & (h.label > 0)).mean()), 4),
        "anomaly_detector_flag_rate_on_policy_normal_rows": round(float(h[h.label == 0].anomaly.mean()), 4)}
    metrics["fault_sessions"] = {}
    for kind in ("overheat", "drag", "battery", "stuck_current"):
        f = ev[(ev.fault_kind == kind) & (ev.fault_active == 1)]
        lat = []
        for sid, g in f.groupby("session"):
            hit = np.flatnonzero((g.status_idx > 0).to_numpy())
            lat.append(hit[0] * C.DT if len(hit) else np.nan)
        metrics["fault_sessions"][kind] = {
            "fraction_of_fault_rows_flagged": round(float((f.status_idx > 0).mean()), 3),
            "median_seconds_to_first_flag": round(float(np.nanmedian(lat)), 1),
            "anomaly_detector_flag_rate": round(float(f.anomaly.mean()), 3),
            "unseen_in_training": kind == "stuck_current"}
    ra = assess_frame(model, real, explain_rows=False)
    ra = ra[ra.status_idx >= 0]
    metrics["real_csv_check"] = {"rows": int(len(ra)),
                                 "fraction_NORMAL": round(float((ra.status_idx == 0).mean()), 4),
                                 "anomaly_detector_flag_rate": round(float(ra.anomaly.mean()), 4)}
    print(json.dumps(metrics, indent=2))

    print("[7/7] Saving")
    out_dir = root / "models"
    model.save(out_dir / "health_model.joblib")
    card = {"model_version": __version__, "trained_utc": model.meta["created"],
            "libraries": {"scikit-learn": sklearn.__version__, "python": platform.python_version()},
            "training_data": {"real_rows": int(len(real)), "synthetic_rows": int(len(syn)),
                              "synthetic_sessions": syn.groupby("fault_kind")["session"].nunique().to_dict()},
            "features": FEATURE_NAMES, "policy_thresholds": C.THRESHOLDS,
            "anomaly_threshold": threshold, "metrics": metrics,
            "caveats": [
                "Labels come from the engineering thresholds in config.py, so classifier scores "
                "measure how well the model reproduces that policy, not real-world fault detection.",
                "Faults are simulated; only ~100 s of real healthy driving was available.",
                "Retrain after collecting more real runs or changing thresholds."]}
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    if not args.no_plots:
        _plots(root / "reports", cm, clf, ev, threshold)
    print(f"      saved {out_dir / 'health_model.joblib'}  ({time.time() - t_start:.0f}s total)")
    return 0


def _plots(rep_dir: Path, cm, clf, ev, threshold) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("      (matplotlib not installed - skipping plots)")
        return
    rep_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), C.LABELS, rotation=20)
    ax.set_yticks(range(3), C.LABELS)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm[i].sum() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("policy label")
    ax.set_title("Cross-validated confusion matrix")
    fig.tight_layout()
    fig.savefig(rep_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    imp = pd.Series(clf.feature_importances_, index=FEATURE_NAMES).sort_values()[-12:]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    imp.plot.barh(ax=ax, color="#2a9d8f")
    ax.set_title("Top feature importances")
    fig.tight_layout()
    fig.savefig(rep_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(ev.anomaly_score.min(), ev.anomaly_score.max(), 60)
    for kind, col in (("none", "#2a9d8f"), ("stuck_current", "#e76f51")):
        s = ev[(ev.fault_kind == kind) & ((ev.fault_active == 1) | (kind == "none"))].anomaly_score
        ax.hist(s.dropna(), bins=bins, alpha=0.65, density=True, color=col,
                label="healthy" if kind == "none" else "unseen fault: stuck current sensor")
    ax.axvline(threshold, color="k", ls="--", label="alarm threshold")
    ax.set_xlabel("anomaly score (lower = stranger)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rep_dir / "anomaly_scores.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
