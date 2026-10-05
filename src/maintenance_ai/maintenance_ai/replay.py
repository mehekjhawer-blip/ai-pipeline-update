"""Run the trained model over a CSV (no ROS needed) - handy for demos and debugging.

    python3 -m maintenance_ai.replay                          # your logger CSV
    python3 -m maintenance_ai.replay --fault drag --onset 40  # overlay a demo fault
    python3 -m maintenance_ai.replay --simulate 120 --fault overheat --out reports/replay.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .engine import assess_frame
from .faults import KINDS, FaultInjector, FaultProfile
from .model import HealthModel
from .paths import data_path, find_repo_root, model_path
from .preprocess import clean
from .simulate import simulate_session


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--simulate", type=float, default=None, metavar="SECONDS",
                    help="use a synthetic healthy drive instead of the CSV")
    ap.add_argument("--fault", choices=KINDS, default=None)
    ap.add_argument("--onset", type=float, default=30.0, help="fault start (s)")
    ap.add_argument("--out", type=Path, default=None, help="write per-row results here")
    args = ap.parse_args(argv)

    model = HealthModel.load(model_path())
    if args.simulate:
        df = simulate_session(np.random.default_rng(3), args.simulate)[C.CSV_COLUMNS]
    else:
        df = pd.read_csv(args.csv or data_path(find_repo_root()))
    df, rep = clean(df)
    print(rep)

    if args.fault:
        inj, rows = FaultInjector(), []
        t0 = df["timestamp"].iloc[0]
        for r in df.to_dict("records"):
            if r["timestamp"] - t0 >= args.onset and inj.profile is None:
                inj.set_fault(FaultProfile.demo(args.fault))
            out = inj.apply(r)
            out["segment"] = r["segment"]
            rows.append(out)
        df = pd.DataFrame(rows)

    res = assess_frame(model, df)
    res.insert(0, "timestamp", df["timestamp"].to_numpy())
    print("\nStatus counts:", res["status"].value_counts().to_dict())
    change = res["status"].ne(res["status"].shift())
    t0 = df["timestamp"].iloc[0]
    print("\nTimeline (first row of each status change):")
    for _, r in res[change].iterrows():
        print(f"  t={r.timestamp - t0:7.1f}s  {r.status:<10} {r.reason}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out, index=False)
        print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
