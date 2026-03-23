from __future__ import annotations

from pathlib import Path

import pandas as pd

KEYS = ["season", "month", "workingday", "hour"]
STAT_RENAME = {
    "sample_n": "stat_sample_n_s_m_w_h",
    "count_mean": "stat_count_mean_s_m_w_h",
    "count_std": "stat_count_std_s_m_w_h",
    "count_q05": "stat_count_q05_s_m_w_h",
    "count_q25": "stat_count_q25_s_m_w_h",
    "count_q50": "stat_count_q50_s_m_w_h",
    "count_q75": "stat_count_q75_s_m_w_h",
    "count_q95": "stat_count_q95_s_m_w_h",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "train_test_merged.csv"
    stats_path = root / "outputs" / "count_quantiles_train_season_month_workingday_hour.csv"
    out_path = root / "data" / "train_test_merged_with_bucket_stats.csv"

    df = pd.read_csv(data_path)
    stats = pd.read_csv(stats_path)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["month"] = df["datetime"].dt.month

    for c in KEYS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        stats[c] = pd.to_numeric(stats[c], errors="coerce")

    merged = df.merge(stats, on=KEYS, how="left", sort=False)
    merged = merged.rename(columns=STAT_RENAME)

    # Keep original order and columns, then append stat columns.
    stat_cols = list(STAT_RENAME.values())
    original_cols = [c for c in df.columns if c != "month"]
    merged = merged[original_cols + stat_cols]

    merged["datetime"] = merged["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    merged.to_csv(out_path, index=False)

    target_sources = ["train", "test", "miss_train", "miss_test"]
    chk = merged[merged["source"].isin(target_sources)]
    nulls = chk[stat_cols].isna().sum().sum()

    print(f"output: {out_path}")
    print(f"rows: {len(merged)}")
    print(f"target_rows: {len(chk)}")
    print(f"target_stat_nulls: {int(nulls)}")


if __name__ == "__main__":
    main()
