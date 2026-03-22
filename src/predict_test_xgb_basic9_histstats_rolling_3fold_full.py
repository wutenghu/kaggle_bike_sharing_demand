from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


BASE_FEATURES = [
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
HIST_FEATURES = [
    "hist_mean_global",
    "hist_mean_hour",
    "hist_mean_dayofweek",
    "hist_mean_month",
    "hist_mean_hour_x_workingday",
    "hist_mean_hour_x_season",
]
ALL_FEATURES = BASE_FEATURES + HIST_FEATURES
TARGET = "count"


def add_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["datetime"].dt.hour
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["month"] = out["datetime"].dt.month
    out["hour_x_workingday"] = out["hour"] * out["workingday"]
    out["hour_x_season"] = out["hour"] * out["season"]
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    models_dir = root / "models"
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_like_path = data_dir / "cv2_random_train_miss_train_20260322_205429.csv"
    test_path = data_dir / "test.csv"
    model_path = models_dir / "xgboost_basic9_histstats_rolling_3fold_full.json"

    train_df = pd.read_csv(train_like_path)
    test_df = pd.read_csv(test_path)

    train_df["datetime"] = pd.to_datetime(train_df["datetime"], errors="coerce")
    test_df["datetime"] = pd.to_datetime(test_df["datetime"], errors="coerce")

    train_df = train_df.dropna(subset=["datetime", TARGET]).copy()
    train_df = train_df.sort_values("datetime").reset_index(drop=True)

    train_df = add_time_keys(train_df)
    test_df = add_time_keys(test_df)

    for c in ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed", TARGET]:
        if c in train_df.columns:
            train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
        if c in test_df.columns:
            test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

    global_mean = float(train_df[TARGET].mean())
    test_df["hist_mean_global"] = global_mean

    mappings = {
        "hist_mean_hour": train_df.groupby("hour")[TARGET].mean(),
        "hist_mean_dayofweek": train_df.groupby("dayofweek")[TARGET].mean(),
        "hist_mean_month": train_df.groupby("month")[TARGET].mean(),
        "hist_mean_hour_x_workingday": train_df.groupby("hour_x_workingday")[TARGET].mean(),
        "hist_mean_hour_x_season": train_df.groupby("hour_x_season")[TARGET].mean(),
    }

    for feat, mp in mappings.items():
        key = feat.replace("hist_mean_", "")
        test_df[feat] = test_df[key].map(mp).fillna(global_mean)

    # Full-model train-time fallback parity: median fill from train feature space
    train_feat_for_fill = pd.DataFrame(index=train_df.index)
    for c in BASE_FEATURES:
        train_feat_for_fill[c] = train_df[c]
    train_feat_for_fill["hist_mean_global"] = train_df[TARGET].expanding().mean().shift(1).fillna(global_mean)
    train_feat_for_fill["hist_mean_hour"] = train_df["hour"].map(mappings["hist_mean_hour"]).fillna(global_mean)
    train_feat_for_fill["hist_mean_dayofweek"] = train_df["dayofweek"].map(mappings["hist_mean_dayofweek"]).fillna(global_mean)
    train_feat_for_fill["hist_mean_month"] = train_df["month"].map(mappings["hist_mean_month"]).fillna(global_mean)
    train_feat_for_fill["hist_mean_hour_x_workingday"] = train_df["hour_x_workingday"].map(mappings["hist_mean_hour_x_workingday"]).fillna(global_mean)
    train_feat_for_fill["hist_mean_hour_x_season"] = train_df["hour_x_season"].map(mappings["hist_mean_hour_x_season"]).fillna(global_mean)

    fill_values = train_feat_for_fill[ALL_FEATURES].median(numeric_only=True).fillna(0)
    x_test = test_df[ALL_FEATURES].fillna(fill_values)

    model = XGBRegressor()
    model.load_model(str(model_path))

    pred = np.expm1(model.predict(x_test))
    pred = np.clip(pred, 0, None)
    pred_int = np.rint(pred).astype(int)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = outputs_dir / f"submission_test_xgb_basic9_histstats_rolling3_full_{ts}.csv"
    sub = pd.DataFrame({
        "datetime": test_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "count": pred_int,
    })
    sub.to_csv(out_path, index=False)

    # Alignment check against test.csv datetime order
    aligned = bool(
        test_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").reset_index(drop=True).equals(
            sub["datetime"].astype(str).reset_index(drop=True)
        )
    )

    print(f"output={out_path}")
    print(f"rows={len(sub)}")
    print(f"count_min={int(sub['count'].min())}")
    print(f"count_max={int(sub['count'].max())}")
    print(f"count_mean={sub['count'].mean():.6f}")
    print(f"aligned_with_test_csv={aligned}")


if __name__ == "__main__":
    main()
