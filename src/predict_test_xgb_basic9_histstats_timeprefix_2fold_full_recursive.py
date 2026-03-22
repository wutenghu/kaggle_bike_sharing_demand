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


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
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

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    model_path = models_dir / "xgboost_basic9_histstats_timeprefix_2fold_full.json"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train["datetime"] = pd.to_datetime(train["datetime"], errors="coerce")
    test["datetime"] = pd.to_datetime(test["datetime"], errors="coerce")

    for c in ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed", TARGET]:
        if c in train.columns:
            train[c] = pd.to_numeric(train[c], errors="coerce")
        if c in test.columns:
            test[c] = pd.to_numeric(test[c], errors="coerce")

    train = train.dropna(subset=["datetime", TARGET]).copy()
    train = add_keys(train)
    test = add_keys(test)

    # model
    model = XGBRegressor()
    model.load_model(str(model_path))

    # initialize running stats from true train counts
    global_sum = float(train[TARGET].sum())
    global_cnt = int(train[TARGET].count())

    key_stats = {}
    for key in ["hour", "dayofweek", "month", "hour_x_workingday", "hour_x_season"]:
        g = train.groupby(key)[TARGET].agg(["sum", "count"])
        key_stats[key] = {
            k: [float(v["sum"]), int(v["count"])]
            for k, v in g.iterrows()
        }

    global_mean = global_sum / max(global_cnt, 1)

    # fill-value parity with training: use train medians of feature frame
    train_feat = pd.DataFrame(index=train.index)
    for c in BASE_FEATURES:
        train_feat[c] = train[c]
    train_feat["hist_mean_global"] = train[TARGET].expanding().mean().shift(1).fillna(global_mean)
    for key, feat in [
        ("hour", "hist_mean_hour"),
        ("dayofweek", "hist_mean_dayofweek"),
        ("month", "hist_mean_month"),
        ("hour_x_workingday", "hist_mean_hour_x_workingday"),
        ("hour_x_season", "hist_mean_hour_x_season"),
    ]:
        grp = train.groupby(key)[TARGET]
        train_feat[feat] = ((grp.cumsum() - train[TARGET]) / grp.cumcount().replace(0, np.nan)).fillna(global_mean)
    fill_values = train_feat[ALL_FEATURES].median(numeric_only=True).fillna(0)

    # recursive prediction in chronological order
    test_sorted = test.sort_values("datetime").copy()
    pred_map = {}

    for idx, row in test_sorted.iterrows():
        feat = {
            "season": row["season"],
            "holiday": row["holiday"],
            "workingday": row["workingday"],
            "weather": row["weather"],
            "temp": row["temp"],
            "atemp": row["atemp"],
            "humidity": row["humidity"],
            "windspeed": row["windspeed"],
            "hour": row["hour"],
            "hist_mean_global": global_sum / max(global_cnt, 1),
        }

        for key, feat_name in [
            ("hour", "hist_mean_hour"),
            ("dayofweek", "hist_mean_dayofweek"),
            ("month", "hist_mean_month"),
            ("hour_x_workingday", "hist_mean_hour_x_workingday"),
            ("hour_x_season", "hist_mean_hour_x_season"),
        ]:
            kval = row[key]
            if pd.isna(kval):
                feat[feat_name] = global_mean
            else:
                rec = key_stats[key].get(kval)
                feat[feat_name] = (rec[0] / rec[1]) if rec and rec[1] > 0 else (global_sum / max(global_cnt, 1))

        x = pd.DataFrame([feat], columns=ALL_FEATURES).fillna(fill_values)
        pred = float(np.expm1(model.predict(x)[0]))
        pred = max(0.0, pred)
        pred_int = int(np.rint(pred))

        pred_map[idx] = pred_int

        # update running stats with predicted count
        global_sum += pred
        global_cnt += 1

        for key in ["hour", "dayofweek", "month", "hour_x_workingday", "hour_x_season"]:
            kval = row[key]
            if pd.isna(kval):
                continue
            if kval not in key_stats[key]:
                key_stats[key][kval] = [0.0, 0]
            key_stats[key][kval][0] += pred
            key_stats[key][kval][1] += 1

    test_out = test.copy()
    test_out["count"] = pd.Series(pred_map)
    test_out["count"] = test_out["count"].astype(int)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = outputs_dir / f"submission_test_xgb_basic9_histstats_timeprefix_2fold_full_recursive_{ts}.csv"
    out = pd.DataFrame({
        "datetime": test_out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "count": test_out["count"],
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
