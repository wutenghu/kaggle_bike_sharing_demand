from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
TARGET = "count"
FOLD_COL = "cv_fold"
STAT_FEATURE = "stat_mean_count_year_season_workingday_hour"


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


def add_base_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["datetime"].dt.year
    out["hour"] = out["datetime"].dt.hour
    return out


def add_group_stat_feature(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    tr = train_df.copy()
    va = valid_df.copy()

    # only source=train rows are used to build statistics
    stat_src = tr[tr["source"] == "train"].copy()
    if stat_src.empty:
        stat_src = tr.copy()

    group_cols = ["year", "season", "workingday", "hour"]
    mapping = stat_src.groupby(group_cols)[TARGET].mean().rename(STAT_FEATURE).reset_index()

    tr = tr.merge(mapping, on=group_cols, how="left")
    va = va.merge(mapping, on=group_cols, how="left")

    fallback = float(stat_src[TARGET].mean()) if len(stat_src) else float(tr[TARGET].mean())
    tr[STAT_FEATURE] = tr[STAT_FEATURE].fillna(fallback)
    va[STAT_FEATURE] = va[STAT_FEATURE].fillna(fallback)

    return tr, va, fallback


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
    df = add_base_time(df)

    required = set(["datetime", "source", TARGET, FOLD_COL] + [c for c in BASE_FEATURES if c not in {"year", "hour"}])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in BASE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    work_df = df.dropna(subset=["datetime", TARGET]).copy()

    folds = sorted(work_df[FOLD_COL].dropna().astype(int).unique().tolist())
    if len(folds) != 2:
        raise ValueError(f"Expect 2 folds, found {folds}")

    best_params, params_from = load_best_basic9_params(outputs_dir)

    features = BASE_FEATURES + [STAT_FEATURE]
    oof_pred = np.full(len(work_df), np.nan, dtype=float)
    fold_metrics: list[FoldMetric] = []
    fold_models: list[str] = []

    for fold in folds:
        tr_idx = work_df.index[work_df[FOLD_COL] != fold]
        va_idx = work_df.index[work_df[FOLD_COL] == fold]

        tr_raw = work_df.loc[tr_idx].copy()
        va_raw = work_df.loc[va_idx].copy()

        tr, va, _ = add_group_stat_feature(tr_raw, va_raw)

        fill_values = tr[features].median(numeric_only=True).fillna(0)
        x_tr = tr[features].fillna(fill_values)
        x_va = va[features].fillna(fill_values)

        y_tr = np.log1p(np.clip(tr[TARGET].to_numpy(dtype=float), 0, None))
        y_va = va[TARGET].to_numpy(dtype=float)

        model = build_model(best_params)
        model.fit(x_tr, y_tr, verbose=False)

        pred = np.expm1(model.predict(x_va))
        pred = np.clip(pred, 0, None)
        oof_pred[va_idx] = pred

        model_name = f"xgboost_basic10_y_s_w_h_stat_2fold_fold{fold}.json"
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

    # full model
    full_df = work_df.copy()
    full_df, _, fallback = add_group_stat_feature(full_df, full_df.iloc[0:0].copy())
    full_df[STAT_FEATURE] = full_df[STAT_FEATURE].fillna(fallback)
    full_fill = full_df[features].median(numeric_only=True).fillna(0)

    full_model = build_model(best_params)
    full_model.fit(
        full_df[features].fillna(full_fill),
        np.log1p(np.clip(full_df[TARGET].to_numpy(dtype=float), 0, None)),
        verbose=False,
    )

    full_model_name = "xgboost_basic10_y_s_w_h_stat_2fold_full.json"
    full_model.save_model(str(models_dir / full_model_name))

    oof = work_df[["datetime", "source", TARGET, FOLD_COL]].copy()
    oof["pred_count"] = oof_pred
    oof_path = outputs_dir / "oof_xgboost_basic10_y_s_w_h_stat_2fold.csv"
    oof.to_csv(oof_path, index=False)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_data": data_path.name,
        "feature_strategy": "base10(year+base9) + mean(count|year,season,workingday,hour) from source=train",
        "base_features": BASE_FEATURES,
        "added_features": [STAT_FEATURE],
        "features": features,
        "target": TARGET,
        "cv_strategy": "random_2fold_from_cv_fold",
        "fold_column": FOLD_COL,
        "group_stat_rule": "statistics computed using only source=train rows in each training fold",
        "fold_metrics": [asdict(m) for m in fold_metrics],
        "fold_scores": [m.rmsle for m in fold_metrics],
        "mean_rmsle": mean_rmsle,
        "std_rmsle": std_rmsle,
        "using_params_from": params_from,
        "model_paths": fold_models + [full_model_name],
        "oof_path": oof_path.name,
    }

    metrics_path = outputs_dir / "xgboost_basic10_y_s_w_h_stat_2fold_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = reports_dir / "xgboost_basic10_y_s_w_h_stat_2fold_report.md"
    report_lines = [
        "# XGBoost Basic10 + YearSeasonWorkingdayHour Stat 2-Fold Report",
        "",
        f"- Time: {payload['timestamp']}",
        f"- Input: `{payload['input_data']}`",
        f"- Feature strategy: `{payload['feature_strategy']}`",
        f"- Added feature: `{STAT_FEATURE}`",
        f"- Group stat rule: `{payload['group_stat_rule']}`",
        f"- CV strategy: `{payload['cv_strategy']}`",
        f"- Fold RMSLE: `{', '.join(f'{x:.6f}' for x in payload['fold_scores'])}`",
        f"- Mean RMSLE: `{mean_rmsle:.6f}`",
        f"- Std RMSLE: `{std_rmsle:.6f}`",
        f"- Models: `{', '.join(payload['model_paths'])}`",
        f"- OOF: `{payload['oof_path']}`",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    print(f"mean_rmsle: {mean_rmsle:.6f}")
    print(f"std_rmsle: {std_rmsle:.6f}")


if __name__ == "__main__":
    main()
