from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

BASE_FEATURES = [
    "year",
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
STAT_FEATURE = "stat_mean_count_year_season_workingday_hour"
TARGET = "count"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    outputs_dir = root / "outputs"
    model_dir = root / "models"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_like_path = data_dir / "cv2_random_train_miss_train_20260322_205429.csv"
    test_path = data_dir / "test.csv"
    model_path = model_dir / "xgboost_basic10_y_s_w_h_stat_2fold_full.json"

    train_like = pd.read_csv(train_like_path)
    test_df = pd.read_csv(test_path)

    train_like["datetime"] = pd.to_datetime(train_like["datetime"], errors="coerce")
    test_df["datetime"] = pd.to_datetime(test_df["datetime"], errors="coerce")

    train_like["year"] = train_like["datetime"].dt.year
    train_like["hour"] = train_like["datetime"].dt.hour
    test_df["year"] = test_df["datetime"].dt.year
    test_df["hour"] = test_df["datetime"].dt.hour

    for c in ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed", "year", "hour", TARGET]:
        if c in train_like.columns:
            train_like[c] = pd.to_numeric(train_like[c], errors="coerce")
        if c in test_df.columns:
            test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

    train_like = train_like.dropna(subset=["datetime", TARGET]).copy()

    # stat from source=train only
    stat_src = train_like[train_like["source"] == "train"].copy()
    if stat_src.empty:
        stat_src = train_like.copy()

    group_cols = ["year", "season", "workingday", "hour"]
    mapping = stat_src.groupby(group_cols)[TARGET].mean().rename(STAT_FEATURE).reset_index()
    test_df = test_df.merge(mapping, on=group_cols, how="left")

    fallback = float(stat_src[TARGET].mean())
    test_df[STAT_FEATURE] = test_df[STAT_FEATURE].fillna(fallback)

    features = BASE_FEATURES + [STAT_FEATURE]

    # fill value parity from full training feature frame
    full_train = train_like.merge(mapping, on=group_cols, how="left")
    full_train[STAT_FEATURE] = full_train[STAT_FEATURE].fillna(fallback)
    fill_values = full_train[features].median(numeric_only=True).fillna(0)

    x_test = test_df[features].fillna(fill_values)

    model = XGBRegressor()
    model.load_model(str(model_path))
    pred = np.expm1(model.predict(x_test))
    pred = np.clip(pred, 0, None)
    pred_int = np.rint(pred).astype(int)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = outputs_dir / f"submission_test_xgb_basic10_y_s_w_h_stat_2fold_full_{ts}.csv"
    out = pd.DataFrame({
        "datetime": test_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "count": pred_int,
    })
    out.to_csv(out_path, index=False)

    aligned = pd.read_csv(test_path)["datetime"].astype(str).equals(out["datetime"].astype(str))

    print(f"output={out_path}")
    print(f"rows={len(out)}")
    print(f"aligned_with_test={aligned}")
    print(f"count_min={int(out['count'].min())}")
    print(f"count_max={int(out['count'].max())}")
    print(f"count_mean={out['count'].mean():.6f}")


if __name__ == "__main__":
    main()
