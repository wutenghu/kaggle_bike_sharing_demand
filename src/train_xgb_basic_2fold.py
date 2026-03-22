from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


FEATURES = [
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
    )


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

    required = set([TARGET, FOLD_COL, "datetime"] + [c for c in FEATURES if c != "hour"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df.copy()
    if "datetime" not in work_df.columns:
        raise ValueError("Missing required column: datetime")
    work_df["datetime"] = pd.to_datetime(work_df["datetime"], errors="coerce")
    work_df["hour"] = work_df["datetime"].dt.hour
    work_df[TARGET] = pd.to_numeric(work_df[TARGET], errors="coerce")
    for c in FEATURES:
        work_df[c] = pd.to_numeric(work_df[c], errors="coerce")
    work_df = work_df.dropna(subset=[TARGET]).reset_index(drop=True)

    oof_pred = np.zeros(len(work_df), dtype=float)
    fold_metrics: list[FoldMetric] = []

    unique_folds = sorted(work_df[FOLD_COL].dropna().astype(int).unique().tolist())
    if len(unique_folds) != 2:
        raise ValueError(f"Expect 2 folds, found: {unique_folds}")

    for fold in unique_folds:
        train_idx = work_df.index[work_df[FOLD_COL] != fold]
        valid_idx = work_df.index[work_df[FOLD_COL] == fold]

        train_df = work_df.loc[train_idx].copy()
        valid_df = work_df.loc[valid_idx].copy()

        fill_values = train_df[FEATURES].median(numeric_only=True)
        x_train = train_df[FEATURES].fillna(fill_values)
        x_valid = valid_df[FEATURES].fillna(fill_values)
        y_train = np.log1p(train_df[TARGET].to_numpy(dtype=float))
        y_valid = valid_df[TARGET].to_numpy(dtype=float)

        model = build_model()
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, np.log1p(np.clip(y_valid, 0, None)))],
            verbose=False,
        )

        pred = np.expm1(model.predict(x_valid))
        pred = np.clip(pred, 0, None)
        oof_pred[valid_idx] = pred

        fold_score = rmsle(y_valid, pred)
        fold_metrics.append(
            FoldMetric(
                fold=int(fold),
                train_size=int(len(train_df)),
                valid_size=int(len(valid_df)),
                rmsle=fold_score,
            )
        )

        model.save_model(str(models_dir / f"xgboost_basic_9feat_hour_fold{fold}.json"))

    mean_rmsle = float(np.mean([m.rmsle for m in fold_metrics]))

    full_fill = work_df[FEATURES].median(numeric_only=True)
    full_model = build_model()
    full_model.fit(
        work_df[FEATURES].fillna(full_fill),
        np.log1p(work_df[TARGET].to_numpy(dtype=float)),
        verbose=False,
    )
    full_model_path = models_dir / "xgboost_basic_9feat_hour_full.json"
    full_model.save_model(str(full_model_path))

    oof = work_df[["datetime", "source", TARGET, FOLD_COL]].copy()
    oof["pred_count"] = oof_pred
    oof_path = outputs_dir / "oof_xgboost_basic_2fold.csv"
    oof.to_csv(oof_path, index=False)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_data": input_path.name,
        "features": FEATURES,
        "target": TARGET,
        "fold_column": FOLD_COL,
        "fold_metrics": [asdict(m) for m in fold_metrics],
        "mean_rmsle": mean_rmsle,
        "model_paths": [
            f"xgboost_basic_9feat_hour_fold{f}.json" for f in unique_folds
        ]
        + [full_model_path.name],
        "oof_path": oof_path.name,
    }
    metrics_path = outputs_dir / "xgboost_basic_2fold_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = reports_dir / "xgboost_basic_2fold_report.md"
    fold_scores_str = ", ".join(f"{m.rmsle:.6f}" for m in fold_metrics)
    report_lines = [
        "# XGBoost Basic 2-Fold CV Report",
        "",
        f"- Time: {payload['timestamp']}",
        f"- Input: `{payload['input_data']}`",
        f"- Features: `{', '.join(FEATURES)}`",
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
