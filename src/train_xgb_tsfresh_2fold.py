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
MAX_TSFRESH_FEATURES = 120
MAX_MISSING_RATIO = 0.40


@dataclass
class FoldMetric:
    fold: int
    train_size: int
    valid_size: int
    rmsle: float
    tsfresh_feature_count: int


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def latest_cv_file(data_dir: Path) -> Path:
    files = sorted(
        data_dir.glob("cv2_random_train_miss_train_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("No cv2_random_train_miss_train_*.csv found in data/")
    return files[0]


def build_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1400,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
    )


def discover_tsfresh_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c.startswith("count__") or c.startswith("count_w"):
            cols.append(c)
    return cols


def select_tsfresh_features(
    train_df: pd.DataFrame,
    candidates: list[str],
    y_log: np.ndarray,
    max_features: int = MAX_TSFRESH_FEATURES,
) -> list[str]:
    if not candidates:
        return []

    x = train_df[candidates].copy()
    for c in candidates:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    miss_ratio = x.isna().mean()
    keep = miss_ratio[miss_ratio <= MAX_MISSING_RATIO].index.tolist()
    if not keep:
        return []

    x = x[keep]
    nunique = x.nunique(dropna=True)
    keep = nunique[nunique > 1].index.tolist()
    if not keep:
        return []

    x = x[keep].fillna(x[keep].median(numeric_only=True))
    y_series = pd.Series(y_log, index=x.index)

    corr = {}
    for c in x.columns:
        v = x[c]
        if v.std(ddof=0) == 0:
            continue
        val = v.corr(y_series)
        if pd.notna(val):
            corr[c] = float(abs(val))

    if not corr:
        return []

    ranked = sorted(corr.items(), key=lambda kv: kv[1], reverse=True)
    return [c for c, _ in ranked[:max_features]]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    outputs_dir = root / "outputs"
    models_dir = root / "models"
    reports_dir = root / "reports"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    input_path = latest_cv_file(data_dir)
    df = pd.read_csv(input_path)

    required = set([TARGET, FOLD_COL, "datetime"] + [c for c in BASE_FEATURES if c != "hour"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df.copy()
    work_df["datetime"] = pd.to_datetime(work_df["datetime"], errors="coerce")
    work_df["hour"] = work_df["datetime"].dt.hour
    work_df[TARGET] = pd.to_numeric(work_df[TARGET], errors="coerce")
    for c in BASE_FEATURES:
        work_df[c] = pd.to_numeric(work_df[c], errors="coerce")
    work_df = work_df.dropna(subset=[TARGET]).reset_index(drop=True)

    tsfresh_candidates = discover_tsfresh_columns(work_df)
    for c in tsfresh_candidates:
        work_df[c] = pd.to_numeric(work_df[c], errors="coerce")

    oof_pred = np.zeros(len(work_df), dtype=float)
    oof_tsfresh_count = np.zeros(len(work_df), dtype=int)
    fold_metrics: list[FoldMetric] = []
    fold_feature_map: dict[str, list[str]] = {}

    unique_folds = sorted(work_df[FOLD_COL].dropna().astype(int).unique().tolist())
    if len(unique_folds) != 2:
        raise ValueError(f"Expect 2 folds, found: {unique_folds}")

    for fold in unique_folds:
        train_idx = work_df.index[work_df[FOLD_COL] != fold]
        valid_idx = work_df.index[work_df[FOLD_COL] == fold]

        train_df = work_df.loc[train_idx].copy()
        valid_df = work_df.loc[valid_idx].copy()
        y_train_raw = train_df[TARGET].to_numpy(dtype=float)
        y_valid = valid_df[TARGET].to_numpy(dtype=float)
        y_train_log = np.log1p(np.clip(y_train_raw, 0, None))

        fold_tsfresh = select_tsfresh_features(train_df, tsfresh_candidates, y_train_log)
        features = BASE_FEATURES + fold_tsfresh
        fold_feature_map[f"fold_{fold}"] = fold_tsfresh

        fill_values = train_df[features].median(numeric_only=True)
        x_train = train_df[features].fillna(fill_values)
        x_valid = valid_df[features].fillna(fill_values)

        model = build_model()
        model.fit(
            x_train,
            y_train_log,
            eval_set=[(x_valid, np.log1p(np.clip(y_valid, 0, None)))],
            verbose=False,
        )

        pred = np.expm1(model.predict(x_valid))
        pred = np.clip(pred, 0, None)
        oof_pred[valid_idx] = pred
        oof_tsfresh_count[valid_idx] = len(fold_tsfresh)

        fold_score = rmsle(y_valid, pred)
        fold_metrics.append(
            FoldMetric(
                fold=int(fold),
                train_size=int(len(train_df)),
                valid_size=int(len(valid_df)),
                rmsle=fold_score,
                tsfresh_feature_count=len(fold_tsfresh),
            )
        )

        model.save_model(str(models_dir / f"xgboost_tsfresh_2fold_fold{fold}.json"))

    mean_rmsle = float(np.mean([m.rmsle for m in fold_metrics]))

    y_all_log = np.log1p(np.clip(work_df[TARGET].to_numpy(dtype=float), 0, None))
    full_tsfresh = select_tsfresh_features(work_df, tsfresh_candidates, y_all_log)
    full_features = BASE_FEATURES + full_tsfresh
    full_fill = work_df[full_features].median(numeric_only=True)
    full_model = build_model()
    full_model.fit(
        work_df[full_features].fillna(full_fill),
        y_all_log,
        verbose=False,
    )
    full_model_path = models_dir / "xgboost_tsfresh_2fold_full.json"
    full_model.save_model(str(full_model_path))

    oof = work_df[["datetime", "source", TARGET, FOLD_COL]].copy()
    oof["pred_count"] = oof_pred
    oof["used_tsfresh_feature_count"] = oof_tsfresh_count
    oof_path = outputs_dir / "oof_xgboost_tsfresh_2fold.csv"
    oof.to_csv(oof_path, index=False)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_data": input_path.name,
        "base_features": BASE_FEATURES,
        "tsfresh_candidate_count": len(tsfresh_candidates),
        "tsfresh_max_features_per_fold": MAX_TSFRESH_FEATURES,
        "target": TARGET,
        "fold_column": FOLD_COL,
        "fold_metrics": [asdict(m) for m in fold_metrics],
        "mean_rmsle": mean_rmsle,
        "fold_selected_tsfresh_features": fold_feature_map,
        "full_selected_tsfresh_features": full_tsfresh,
        "model_paths": [
            f"xgboost_tsfresh_2fold_fold{f}.json" for f in unique_folds
        ] + [full_model_path.name],
        "oof_path": oof_path.name,
    }
    metrics_path = outputs_dir / "xgboost_tsfresh_2fold_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = reports_dir / "xgboost_tsfresh_2fold_report.md"
    fold_scores_str = ", ".join(f"{m.rmsle:.6f}" for m in fold_metrics)
    fold_tsfresh_cnt_str = ", ".join(str(m.tsfresh_feature_count) for m in fold_metrics)
    report_lines = [
        "# XGBoost TSFresh 2-Fold CV Report",
        "",
        f"- Time: {payload['timestamp']}",
        f"- Input: `{payload['input_data']}`",
        f"- Base features: `{', '.join(BASE_FEATURES)}`",
        f"- tsfresh candidates: `{len(tsfresh_candidates)}`",
        f"- tsfresh selected per fold: `{fold_tsfresh_cnt_str}`",
        f"- Fold scores (RMSLE): `{fold_scores_str}`",
        f"- Mean RMSLE: `{mean_rmsle:.6f}`",
        f"- Models: `{', '.join(payload['model_paths'])}`",
        f"- OOF: `{payload['oof_path']}`",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    print(f"mean_rmsle: {mean_rmsle:.6f}")


if __name__ == "__main__":
    main()
