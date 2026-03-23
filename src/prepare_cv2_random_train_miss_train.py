from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "hour",
]
TARGET = "count"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    input_path = data_dir / "train_test_merged.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}")

    df = pd.read_csv(input_path)
    required = {"datetime", "source", TARGET, *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[df["source"].isin(["train", "miss_train"])].copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work = work.dropna(subset=["datetime"]).sort_values("datetime", kind="stable").reset_index(drop=True)

    for c in FEATURES + [TARGET]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=[TARGET]).reset_index(drop=True)

    rng = np.random.RandomState(42)
    work["cv_fold"] = rng.randint(0, 2, size=len(work))

    out = work[["datetime", "source", *FEATURES, TARGET, "cv_fold"]].copy()
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = data_dir / f"cv2_random_train_miss_train_{ts}.csv"
    out.to_csv(out_path, index=False)

    counts = out["cv_fold"].value_counts().to_dict()
    print(f"output: {out_path}")
    print(f"rows: {len(out)}")
    print(f"fold_counts: {counts}")


if __name__ == "__main__":
    main()
