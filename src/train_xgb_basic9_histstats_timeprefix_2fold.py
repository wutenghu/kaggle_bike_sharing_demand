from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
TARGET = "count"
FOLD_COL = "cv_fold"


@dataclass
class FoldMetric:
    fold: int
    train_size: int
    valid_size: int
    rmsle: float


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def build_model(model_params: dict | None = None) -> XGBRegressor:
    if model_params:
        return XGBRegressor(**model_params)
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        tree_method="hist",
    )


def load_best_basic9_params(outputs_dir: Path) -> tuple[dict | None, str | None]:
    metrics_path = outputs_dir / "xgboost_basic9_bayes_2fold_metrics.json"
    if not metrics_path.exists():
        return None, None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return payload.get("best_params"), metrics_path.name


def add_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["datetime"].dt.hour
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["month"] = out["datetime"].dt.month
    out["hour_x_workingday"] = out["hour"] * out["workingday"]
    out["hour_x_season"] = out["hour"] * out["season"]
    return out


def train_past_mean_feature(train_sorted: pd.DataFrame, key: str, feat_name: str) -> pd.Series:
    grp = train_sorted.groupby(key)[TARGET]
    cumsum_excl = grp.cumsum() - train_sorted[TARGET]
    cnt_before = grp.cumcount().replace(0, np.nan)
    return (cumsum_excl / cnt_before).rename(feat_name)


def valid_prefix_mean_feature(
    train_sorted: pd.DataFrame,
    valid_df: pd.DataFrame,
    key: str,
    feat_name: str,
) -> pd.Series:
    # Group-wise prefix lookup using searchsorted (strictly datetime < current row).
    out = pd.Series(np.nan, index=valid_df.index, name=feat_name, dtype=float)
    hist = train_sorted[["datetime", key, TARGET]].copy()
    val_all = valid_df[["datetime", key]].copy()

    for key_val, val_grp in val_all.groupby(key, dropna=False):
        hist_grp = hist[hist[key] == key_val].copy()
        if hist_grp.empty:
            continue
        hist_grp = hist_grp.sort_values("datetime")
        hist_grp["cum_sum"] = hist_grp[TARGET].cumsum()
        hist_grp["cum_cnt"] = np.arange(1, len(hist_grp) + 1)
        hist_state = (
            hist_grp.groupby("datetime", as_index=False)[["cum_sum", "cum_cnt"]]
            .last()
            .sort_values("datetime")
        )
        hist_times = hist_state["datetime"].to_numpy()
        hist_sum = hist_state["cum_sum"].to_numpy(dtype=float)
        hist_cnt = hist_state["cum_cnt"].to_numpy(dtype=float)

        val_grp_sorted = val_grp.sort_values("datetime")
        val_times = val_grp_sorted["datetime"].to_numpy()
        pos = np.searchsorted(hist_times, val_times, side="left") - 1

        vals = np.full(len(val_grp_sorted), np.nan, dtype=float)
        ok = pos >= 0
        vals[ok] = hist_sum[pos[ok]] / hist_cnt[pos[ok]]
        out.loc[val_grp_sorted.index] = vals

    return out


def build_hist_features_timeprefix(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    tr = train_df.copy()
    va = valid_df.copy()

    tr = tr.sort_values("datetime").reset_index(drop=True)

    global_mean = float(tr[TARGET].mean())

    keys = [
        ("hour", "hist_mean_hour"),
        ("dayofweek", "hist_mean_dayofweek"),
        ("month", "hist_mean_month"),
        ("hour_x_workingday", "hist_mean_hour_x_workingday"),
        ("hour_x_season", "hist_mean_hour_x_season"),
    ]

    # Train-side strict past-only global mean
    tr["hist_mean_global"] = tr[TARGET].expanding().mean().shift(1)

    # Valid-side strict prefix global mean from train timeline
    g_hist = tr[["datetime", TARGET]].copy().sort_values("datetime")
    g_hist["cum_sum"] = g_hist[TARGET].cumsum()
    g_hist["cum_cnt"] = np.arange(1, len(g_hist) + 1)
    g_state = (
        g_hist.groupby("datetime", as_index=False)[["cum_sum", "cum_cnt"]]
        .last()
        .sort_values("datetime")
    )
    va_tmp = va[["datetime"]].copy().sort_values("datetime")
    va_g = pd.merge_asof(
        va_tmp,
        g_state,
        on="datetime",
        direction="backward",
        allow_exact_matches=False,
    )
    va_global = (va_g["cum_sum"] / va_g["cum_cnt"]).rename("hist_mean_global")
    va_global.index = va_tmp.index
    va["hist_mean_global"] = va_global.reindex(va.index)

    added = ["hist_mean_global"]

    for key, feat_name in keys:
        tr[feat_name] = train_past_mean_feature(tr, key, feat_name)
        va[feat_name] = valid_prefix_mean_feature(tr, va, key, feat_name)
        added.append(feat_name)

    for c in added:
        tr[c] = tr[c].fillna(global_mean)
        va[c] = va[c].fillna(global_mean)

    return tr, va, added


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "cv2_random_train_miss_train_20260322_205429.csv"
    outputs_dir = root / "outputs"
    models_dir = root / "models"
    reports_dir = root / "reports"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    required = set(["datetime", TARGET, FOLD_COL] + [c for c in BASE_FEATURES if c != "hour"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in [c for c in BASE_FEATURES if c != "hour"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    work_df = df.dropna(subset=["datetime", TARGET]).copy()
    work_df = add_time_keys(work_df)

    best_params, params_from = load_best_basic9_params(outputs_dir)

    folds = sorted(work_df[FOLD_COL].dropna().astype(int).unique().tolist())
    if len(folds) != 2:
        raise ValueError(f"Expect 2 folds, found {folds}")

    oof_pred = np.full(len(work_df), np.nan, dtype=float)
    fold_metrics: list[FoldMetric] = []
    fold_models: list[str] = []
    added_features: list[str] | None = None

    for fold in folds:
        tr_idx = work_df.index[work_df[FOLD_COL] != fold]
        va_idx = work_df.index[work_df[FOLD_COL] == fold]

        tr_raw = work_df.loc[tr_idx].copy()
        va_raw = work_df.loc[va_idx].copy()

        tr_feat, va_feat, added = build_hist_features_timeprefix(tr_raw, va_raw)
        added_features = added
        features = BASE_FEATURES + added

        fill_values = tr_feat[features].median(numeric_only=True).fillna(0)
        x_tr = tr_feat[features].fillna(fill_values)
        x_va = va_feat[features].fillna(fill_values)

        y_tr = np.log1p(np.clip(tr_feat[TARGET].to_numpy(dtype=float), 0, None))
        y_va = va_feat[TARGET].to_numpy(dtype=float)

        model = build_model(best_params)
        model.fit(x_tr, y_tr, verbose=False)

        pred = np.expm1(model.predict(x_va))
        pred = np.clip(pred, 0, None)
        oof_pred[va_idx] = pred

        model_name = f"xgboost_basic9_histstats_timeprefix_2fold_fold{fold}.json"
        model.save_model(str(models_dir / model_name))
        fold_models.append(model_name)

        fold_metrics.append(
            FoldMetric(
                fold=int(fold),
                train_size=int(len(tr_idx)),
                valid_size=int(len(va_idx)),
                rmsle=rmsle(y_va, pred),
            )
        )

    mean_rmsle = float(np.mean([m.rmsle for m in fold_metrics]))
    std_rmsle = float(np.std([m.rmsle for m in fold_metrics]))

    # Full training model for downstream prediction
    full_df = work_df.sort_values("datetime").reset_index(drop=True)
    global_mean = float(full_df[TARGET].mean())
    full_df["hist_mean_global"] = full_df[TARGET].expanding().mean().shift(1).fillna(global_mean)
    for key, feat_name in [
        ("hour", "hist_mean_hour"),
        ("dayofweek", "hist_mean_dayofweek"),
        ("month", "hist_mean_month"),
        ("hour_x_workingday", "hist_mean_hour_x_workingday"),
        ("hour_x_season", "hist_mean_hour_x_season"),
    ]:
        full_df[feat_name] = train_past_mean_feature(full_df, key, feat_name).fillna(global_mean)

    hist_cols = [
        "hist_mean_global",
        "hist_mean_hour",
        "hist_mean_dayofweek",
        "hist_mean_month",
        "hist_mean_hour_x_workingday",
        "hist_mean_hour_x_season",
    ]
    full_features = BASE_FEATURES + hist_cols
    full_fill = full_df[full_features].median(numeric_only=True).fillna(0)
    full_model = build_model(best_params)
    full_model.fit(
        full_df[full_features].fillna(full_fill),
        np.log1p(np.clip(full_df[TARGET].to_numpy(dtype=float), 0, None)),
        verbose=False,
    )

    full_model_name = "xgboost_basic9_histstats_timeprefix_2fold_full.json"
    full_model.save_model(str(models_dir / full_model_name))

    oof = work_df[["datetime", "source", TARGET, FOLD_COL]].copy()
    oof["pred_count"] = oof_pred
    oof_path = outputs_dir / "oof_xgboost_basic9_histstats_timeprefix_2fold.csv"
    oof.to_csv(oof_path, index=False)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_data": data_path.name,
        "feature_strategy": "base9 + leakage_safe_historical_priors_timeprefix",
        "base_features": BASE_FEATURES,
        "added_features": added_features,
        "features": BASE_FEATURES + (added_features or []),
        "target": TARGET,
        "cv_strategy": "random_2fold_from_cv_fold",
        "fold_column": FOLD_COL,
        "hist_feature_rule": "train uses past-only stats; valid uses train-only prefix stats (datetime strict)",
        "fold_metrics": [asdict(m) for m in fold_metrics],
        "fold_scores": [m.rmsle for m in fold_metrics],
        "mean_rmsle": mean_rmsle,
        "std_rmsle": std_rmsle,
        "using_params_from": params_from,
        "model_paths": fold_models + [full_model_name],
        "oof_path": oof_path.name,
    }

    metrics_path = outputs_dir / "xgboost_basic9_histstats_timeprefix_2fold_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# XGBoost Basic9 HistStats TimePrefix 2-Fold Report",
        "",
        f"- Time: {payload['timestamp']}",
        f"- Input: `{payload['input_data']}`",
        f"- Feature strategy: `{payload['feature_strategy']}`",
        f"- Added features: `{', '.join(payload['added_features'] or [])}`",
        f"- CV strategy: `{payload['cv_strategy']}`",
        f"- Hist rule: `{payload['hist_feature_rule']}`",
        f"- Fold RMSLE: `{', '.join(f'{x:.6f}' for x in payload['fold_scores'])}`",
        f"- Mean RMSLE: `{mean_rmsle:.6f}`",
        f"- Std RMSLE: `{std_rmsle:.6f}`",
        f"- Models: `{', '.join(payload['model_paths'])}`",
        f"- OOF: `{payload['oof_path']}`",
        "",
    ]
    report_path = reports_dir / "xgboost_basic9_histstats_timeprefix_2fold_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    print(f"mean_rmsle: {mean_rmsle:.6f}")
    print(f"std_rmsle: {std_rmsle:.6f}")


if __name__ == "__main__":
    main()
