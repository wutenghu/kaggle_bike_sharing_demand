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


@dataclass
class FoldMetric:
    fold: int
    train_size: int
    valid_size: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
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
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["month"] = out["datetime"].dt.month
    out["hour_x_workingday"] = out["hour"] * out["workingday"]
    out["hour_x_season"] = out["hour"] * out["season"]
    return out


def past_mean_train_only(tr: pd.DataFrame, key_col: str, target_col: str) -> pd.Series:
    # Strictly use historical rows before current timestamp inside training fold.
    grp = tr.groupby(key_col)[target_col]
    cumsum_excl_self = grp.cumsum() - tr[target_col]
    cnt_before = grp.cumcount()
    denom = cnt_before.replace(0, np.nan)
    return cumsum_excl_self / denom


def build_hist_features(tr: pd.DataFrame, va: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    tr = tr.copy()
    va = va.copy()

    global_mean = float(tr[TARGET].mean())

    hist_defs = [
        ("hour", "hist_mean_hour"),
        ("dayofweek", "hist_mean_dayofweek"),
        ("month", "hist_mean_month"),
        ("hour_x_workingday", "hist_mean_hour_x_workingday"),
        ("hour_x_season", "hist_mean_hour_x_season"),
    ]

    # Global past mean (causal in train; fixed from train for valid)
    tr["hist_mean_global"] = tr[TARGET].expanding().mean().shift(1)
    va["hist_mean_global"] = global_mean

    added = ["hist_mean_global"]

    for key, feat_name in hist_defs:
        tr[feat_name] = past_mean_train_only(tr, key, TARGET)
        mapping = tr.groupby(key)[TARGET].mean()
        va[feat_name] = va[key].map(mapping)
        added.append(feat_name)

    # Fallback: unseen/early groups use global mean
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
    df["hour"] = df["datetime"].dt.hour

    for c in BASE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    work_df = df.dropna(subset=["datetime", TARGET]).copy()
    work_df = work_df.sort_values("datetime").reset_index(drop=True)
    work_df = add_time_keys(work_df)

    n = len(work_df)
    if n < 8:
        raise ValueError("Not enough samples for rolling 3-fold CV.")

    block = n // 4
    split1 = block
    split2 = block * 2
    split3 = block * 3
    folds = [
        (0, np.arange(0, split1), np.arange(split1, split2)),
        (1, np.arange(0, split2), np.arange(split2, split3)),
        (2, np.arange(0, split3), np.arange(split3, n)),
    ]

    best_params, params_from = load_best_basic9_params(outputs_dir)

    oof_pred = np.full(n, np.nan, dtype=float)
    fold_metrics: list[FoldMetric] = []
    fold_models: list[str] = []
    hist_feature_names: list[str] | None = None

    for fold_id, train_idx, valid_idx in folds:
        tr = work_df.iloc[train_idx].copy()
        va = work_df.iloc[valid_idx].copy()

        tr, va, hist_features = build_hist_features(tr, va)
        hist_feature_names = hist_features
        features = BASE_FEATURES + hist_features

        fill_values = tr[features].median(numeric_only=True).fillna(0)
        x_tr = tr[features].fillna(fill_values)
        x_va = va[features].fillna(fill_values)
        y_tr = np.log1p(np.clip(tr[TARGET].to_numpy(dtype=float), 0, None))
        y_va = va[TARGET].to_numpy(dtype=float)

        model = build_model(best_params)
        model.fit(x_tr, y_tr, verbose=False)

        pred = np.expm1(model.predict(x_va))
        pred = np.clip(pred, 0, None)
        oof_pred[valid_idx] = pred

        model_name = f"xgboost_basic9_histstats_rolling_3fold_fold{fold_id}.json"
        model.save_model(str(models_dir / model_name))
        fold_models.append(model_name)

        fold_metrics.append(
            FoldMetric(
                fold=fold_id,
                train_size=int(len(train_idx)),
                valid_size=int(len(valid_idx)),
                train_start=str(tr["datetime"].min()),
                train_end=str(tr["datetime"].max()),
                valid_start=str(va["datetime"].min()),
                valid_end=str(va["datetime"].max()),
                rmsle=rmsle(y_va, pred),
            )
        )

    mean_rmsle = float(np.mean([m.rmsle for m in fold_metrics]))
    std_rmsle = float(np.std([m.rmsle for m in fold_metrics]))

    # Full model: use causal historical means built on full sorted data.
    full_df = work_df.copy()
    full_df["hist_mean_global"] = full_df[TARGET].expanding().mean().shift(1)
    global_mean_full = float(full_df[TARGET].mean())
    for key, feat_name in [
        ("hour", "hist_mean_hour"),
        ("dayofweek", "hist_mean_dayofweek"),
        ("month", "hist_mean_month"),
        ("hour_x_workingday", "hist_mean_hour_x_workingday"),
        ("hour_x_season", "hist_mean_hour_x_season"),
    ]:
        full_df[feat_name] = past_mean_train_only(full_df, key, TARGET)

    hist_features = [
        "hist_mean_global",
        "hist_mean_hour",
        "hist_mean_dayofweek",
        "hist_mean_month",
        "hist_mean_hour_x_workingday",
        "hist_mean_hour_x_season",
    ]
    for c in hist_features:
        full_df[c] = full_df[c].fillna(global_mean_full)

    full_features = BASE_FEATURES + hist_features
    full_fill = full_df[full_features].median(numeric_only=True).fillna(0)
    full_model = build_model(best_params)
    full_model.fit(
        full_df[full_features].fillna(full_fill),
        np.log1p(np.clip(full_df[TARGET].to_numpy(dtype=float), 0, None)),
        verbose=False,
    )

    full_model_name = "xgboost_basic9_histstats_rolling_3fold_full.json"
    full_model.save_model(str(models_dir / full_model_name))

    oof = work_df[["datetime", "source", TARGET]].copy()
    oof["pred_count"] = oof_pred
    oof_path = outputs_dir / "oof_xgboost_basic9_histstats_rolling_3fold.csv"
    oof.to_csv(oof_path, index=False)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_data": data_path.name,
        "feature_strategy": "base9 + leakage_safe_historical_mean_features",
        "base_features": BASE_FEATURES,
        "added_features": hist_feature_names,
        "features": BASE_FEATURES + (hist_feature_names or []),
        "target": TARGET,
        "cv_strategy": "time_rolling_expanding_window_3fold",
        "fold_definition": "split sorted samples into 4 contiguous blocks; folds validate on block 2/3/4",
        "hist_feature_rule": "train rows use past-only stats; valid rows map stats from train segment only",
        "fold_metrics": [asdict(m) for m in fold_metrics],
        "fold_scores": [m.rmsle for m in fold_metrics],
        "mean_rmsle": mean_rmsle,
        "std_rmsle": std_rmsle,
        "using_params_from": params_from,
        "model_paths": fold_models + [full_model_name],
        "oof_path": oof_path.name,
    }

    metrics_path = outputs_dir / "xgboost_basic9_histstats_rolling_3fold_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# XGBoost Basic9 HistStats Rolling 3-Fold Report",
        "",
        f"- Time: {payload['timestamp']}",
        f"- Input: `{payload['input_data']}`",
        f"- Feature strategy: `{payload['feature_strategy']}`",
        f"- Added features: `{', '.join(payload['added_features'] or [])}`",
        f"- Hist rule: `{payload['hist_feature_rule']}`",
        f"- CV strategy: `{payload['cv_strategy']}`",
        f"- Fold definition: `{payload['fold_definition']}`",
        f"- Fold RMSLE: `{', '.join(f'{x:.6f}' for x in payload['fold_scores'])}`",
        f"- Mean RMSLE: `{mean_rmsle:.6f}`",
        f"- Std RMSLE: `{std_rmsle:.6f}`",
        f"- Params source: `{params_from}`",
        f"- Models: `{', '.join(payload['model_paths'])}`",
        f"- OOF: `{payload['oof_path']}`",
        "",
        "## Fold Windows",
        "",
    ]
    for m in fold_metrics:
        report_lines.append(
            f"- fold{m.fold}: train=[{m.train_start} ~ {m.train_end}] ({m.train_size}), "
            f"valid=[{m.valid_start} ~ {m.valid_end}] ({m.valid_size}), rmsle={m.rmsle:.6f}"
        )
    report_lines.append("")

    report_path = reports_dir / "xgboost_basic9_histstats_rolling_3fold_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    print(f"mean_rmsle: {mean_rmsle:.6f}")
    print(f"std_rmsle: {std_rmsle:.6f}")


if __name__ == "__main__":
    main()
