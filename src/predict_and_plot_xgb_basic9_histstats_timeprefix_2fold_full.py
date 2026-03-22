from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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


def build_train_feature_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    full_df = train_df.sort_values("datetime").reset_index(drop=True).copy()
    global_mean = float(full_df[TARGET].mean())

    full_df["hist_mean_global"] = full_df[TARGET].expanding().mean().shift(1).fillna(global_mean)

    for key, feat_name in [
        ("hour", "hist_mean_hour"),
        ("dayofweek", "hist_mean_dayofweek"),
        ("month", "hist_mean_month"),
        ("hour_x_workingday", "hist_mean_hour_x_workingday"),
        ("hour_x_season", "hist_mean_hour_x_season"),
    ]:
        grp = full_df.groupby(key)[TARGET]
        full_df[feat_name] = ((grp.cumsum() - full_df[TARGET]) / grp.cumcount().replace(0, np.nan)).fillna(global_mean)

    return full_df


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    models_dir = root / "models"
    outputs_dir = root / "outputs"
    figures_dir = root / "figures"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_snapshot_path = data_dir / "cv2_random_train_miss_train_20260322_205429.csv"
    train_real_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    model_path = models_dir / "xgboost_basic9_histstats_timeprefix_2fold_full.json"

    train_snap = pd.read_csv(train_snapshot_path)
    train_real = pd.read_csv(train_real_path)
    test_df = pd.read_csv(test_path)

    train_snap["datetime"] = pd.to_datetime(train_snap["datetime"], errors="coerce")
    train_real["datetime"] = pd.to_datetime(train_real["datetime"], errors="coerce")
    test_df["datetime"] = pd.to_datetime(test_df["datetime"], errors="coerce")

    train_snap[TARGET] = pd.to_numeric(train_snap[TARGET], errors="coerce")
    train_snap = train_snap.dropna(subset=["datetime", TARGET]).copy()

    for c in ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed"]:
        train_snap[c] = pd.to_numeric(train_snap[c], errors="coerce")
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

    train_snap = add_time_keys(train_snap)
    test_df = add_time_keys(test_df)

    # Build mappings from full available train snapshot.
    global_mean = float(train_snap[TARGET].mean())
    test_df["hist_mean_global"] = global_mean

    mappings = {
        "hist_mean_hour": train_snap.groupby("hour")[TARGET].mean(),
        "hist_mean_dayofweek": train_snap.groupby("dayofweek")[TARGET].mean(),
        "hist_mean_month": train_snap.groupby("month")[TARGET].mean(),
        "hist_mean_hour_x_workingday": train_snap.groupby("hour_x_workingday")[TARGET].mean(),
        "hist_mean_hour_x_season": train_snap.groupby("hour_x_season")[TARGET].mean(),
    }

    for feat, mp in mappings.items():
        key = feat.replace("hist_mean_", "")
        test_df[feat] = test_df[key].map(mp).fillna(global_mean)

    # Fill values aligned with full-model training feature preparation.
    train_feat = build_train_feature_frame(train_snap)
    fill_values = train_feat[ALL_FEATURES].median(numeric_only=True).fillna(0)
    x_test = test_df[ALL_FEATURES].fillna(fill_values)

    model = XGBRegressor()
    model.load_model(str(model_path))
    pred = np.expm1(model.predict(x_test))
    pred = np.clip(pred, 0, None)
    pred_int = np.rint(pred).astype(int)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = outputs_dir / f"submission_test_xgb_basic9_histstats_timeprefix_2fold_full_{ts}.csv"

    sub = pd.DataFrame(
        {
            "datetime": test_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "count": pred_int,
        }
    )
    sub.to_csv(sub_path, index=False)

    # Plot daily aggregated train(real) vs test(pred)
    train_real[TARGET] = pd.to_numeric(train_real[TARGET], errors="coerce")
    train_daily = (
        train_real.dropna(subset=["datetime", TARGET])
        .assign(date=lambda d: d["datetime"].dt.floor("D"))
        .groupby("date", as_index=False)[TARGET]
        .sum()
    )

    test_daily = (
        pd.DataFrame({"datetime": test_df["datetime"], TARGET: pred_int})
        .assign(date=lambda d: d["datetime"].dt.floor("D"))
        .groupby("date", as_index=False)[TARGET]
        .sum()
    )

    fig_path = figures_dir / f"train_real_vs_test_pred_daily_xgb_basic9_histstats_timeprefix_2fold_full_{ts}.png"
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_daily["date"], train_daily[TARGET], color="#1f77b4", linewidth=1.8, label="train (real)")
    ax.plot(test_daily["date"], test_daily[TARGET], color="#d62728", linewidth=1.8, label="test (pred)")
    ax.set_title("Daily Count: Train Real vs Test Predicted (HistStats TimePrefix 2-Fold Full)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count per Day")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print(f"submission={sub_path}")
    print(f"figure={fig_path}")
    print(f"rows_test={len(sub)}")


if __name__ == "__main__":
    main()
