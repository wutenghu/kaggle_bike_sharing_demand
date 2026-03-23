from __future__ import annotations

from pathlib import Path

import pandas as pd

KEYS = ["season", "month", "workingday", "hour"]
TARGET = "count"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "train_test_merged.csv"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "count_quantiles_train_season_month_workingday_hour.csv"

    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["source"].eq("train")].copy()
    df["month"] = df["datetime"].dt.month

    for c in KEYS + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=KEYS + [TARGET]).reset_index(drop=True)

    grouped = df.groupby(KEYS, dropna=False)[TARGET]

    stats = grouped.agg(
        sample_n="size",
        count_mean="mean",
        count_std=lambda s: s.std(ddof=0),
    ).reset_index()

    q = grouped.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).unstack(level=-1)
    q.columns = ["count_q05", "count_q25", "count_q50", "count_q75", "count_q95"]
    q = q.reset_index()

    out = stats.merge(q, on=KEYS, how="left")
    out = out[[
        "season",
        "month",
        "workingday",
        "hour",
        "sample_n",
        "count_mean",
        "count_std",
        "count_q05",
        "count_q25",
        "count_q50",
        "count_q75",
        "count_q95",
    ]]

    out.to_csv(out_path, index=False)

    print(f"output: {out_path}")
    print(f"rows: {len(out)}")
    print(f"null_count_std: {int(out['count_std'].isna().sum())}")


if __name__ == "__main__":
    main()
