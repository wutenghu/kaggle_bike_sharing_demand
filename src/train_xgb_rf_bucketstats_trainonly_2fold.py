from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

FEATURES = [
    "season",
    "month",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "hour",
    "stat_count_mean_s_m_w_h",
    "stat_count_std_s_m_w_h",
    "stat_count_q05_s_m_w_h",
    "stat_count_q25_s_m_w_h",
    "stat_count_q50_s_m_w_h",
    "stat_count_q75_s_m_w_h",
    "stat_count_q95_s_m_w_h",
]
TARGET = "count"
FOLD_COL = "cv_fold"
INPUT_NAME = "cv2_random_train_miss_train_with_bucket_stats_20260323_174324.csv"


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


def build_xgb() -> XGBRegressor:
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


def build_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )


def prepare_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df[df["source"].eq("train")].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    if "month" not in df.columns:
        df["month"] = df["datetime"].dt.month

    required = {"datetime", "source", TARGET, FOLD_COL, *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in FEATURES + [TARGET, FOLD_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["datetime", TARGET, FOLD_COL]).reset_index(drop=True)
    return df


def run_cv(df: pd.DataFrame, model_name: str, model_builder):
    oof_pred = np.zeros(len(df), dtype=float)
    fold_metrics: list[FoldMetric] = []

    unique_folds = sorted(df[FOLD_COL].dropna().astype(int).unique().tolist())
    if len(unique_folds) != 2:
        raise ValueError(f"Expect 2 folds, found: {unique_folds}")

    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    fold_model_files: list[str] = []

    for fold in unique_folds:
        train_idx = df.index[df[FOLD_COL] != fold]
        valid_idx = df.index[df[FOLD_COL] == fold]

        train_df = df.loc[train_idx].copy()
        valid_df = df.loc[valid_idx].copy()

        fill_values = train_df[FEATURES].median(numeric_only=True).fillna(0)
        x_train = train_df[FEATURES].fillna(fill_values)
        x_valid = valid_df[FEATURES].fillna(fill_values)

        y_train = np.log1p(train_df[TARGET].to_numpy(dtype=float))
        y_valid = valid_df[TARGET].to_numpy(dtype=float)

        model = model_builder()
        if model_name == "xgb":
            model.fit(x_train, y_train, eval_set=[(x_valid, np.log1p(np.clip(y_valid, 0, None)))], verbose=False)
            pred = np.expm1(model.predict(x_valid))
            model_path = models_dir / f"xgboost_bucketstats_trainonly_2fold_fold{fold}.json"
            model.save_model(str(model_path))
        else:
            model.fit(x_train, y_train)
            pred = np.expm1(model.predict(x_valid))
            model_path = models_dir / f"randomforest_bucketstats_trainonly_2fold_fold{fold}.pkl"
            joblib.dump(model, model_path)

        pred = np.clip(pred, 0, None)
        oof_pred[valid_idx] = pred

        fold_metrics.append(
            FoldMetric(
                fold=int(fold),
                train_size=int(len(train_df)),
                valid_size=int(len(valid_df)),
                rmsle=rmsle(y_valid, pred),
            )
        )
        fold_model_files.append(model_path.name)

    mean_rmsle = float(np.mean([m.rmsle for m in fold_metrics]))

    full_fill = df[FEATURES].median(numeric_only=True).fillna(0)
    x_full = df[FEATURES].fillna(full_fill)
    y_full = np.log1p(df[TARGET].to_numpy(dtype=float))

    full_model = model_builder()
    if model_name == "xgb":
        full_model.fit(x_full, y_full, verbose=False)
        full_json = models_dir / "xgboost_bucketstats_trainonly_2fold_full.json"
        full_pkl = models_dir / "xgboost_bucketstats_trainonly_2fold_full.pkl"
        full_model.save_model(str(full_json))
        joblib.dump(full_model, full_pkl)
        full_files = [full_json.name, full_pkl.name]
    else:
        full_model.fit(x_full, y_full)
        full_pkl = models_dir / "randomforest_bucketstats_trainonly_2fold_full.pkl"
        joblib.dump(full_model, full_pkl)
        full_files = [full_pkl.name]

    return oof_pred, fold_metrics, mean_rmsle, fold_model_files + full_files


def write_outputs(df: pd.DataFrame, oof_pred: np.ndarray, fold_metrics: list[FoldMetric], mean_rmsle: float, model_files: list[str], model_name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    outputs_dir = root / "outputs"
    reports_dir = root / "reports"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    prefix = "xgboost" if model_name == "xgb" else "randomforest"
    metrics_name = f"{prefix}_bucketstats_trainonly_2fold_metrics.json"
    report_name = f"{prefix}_bucketstats_trainonly_2fold_report.md"
    oof_name = f"oof_{prefix}_bucketstats_trainonly_2fold.csv"

    oof = df[["datetime", "source", TARGET, FOLD_COL]].copy()
    oof["pred_count"] = oof_pred
    oof.to_csv(outputs_dir / oof_name, index=False)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_data": INPUT_NAME,
        "training_filter": "source=train",
        "features": FEATURES,
        "target": TARGET,
        "fold_column": FOLD_COL,
        "fold_metrics": [asdict(m) for m in fold_metrics],
        "mean_rmsle": mean_rmsle,
        "model_paths": model_files,
        "oof_path": oof_name,
    }

    (outputs_dir / metrics_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fold_scores = ", ".join(f"{m.rmsle:.6f}" for m in fold_metrics)
    title = "XGBoost" if model_name == "xgb" else "RandomForest"
    lines = [
        f"# {title} BucketStats TrainOnly 2-Fold CV Report",
        "",
        f"- Time: {payload['timestamp']}",
        f"- Input: `{payload['input_data']}`",
        f"- Training filter: `{payload['training_filter']}`",
        f"- Features: `{', '.join(FEATURES)}`",
        f"- Fold scores (RMSLE): `{fold_scores}`",
        f"- Mean RMSLE: `{mean_rmsle:.6f}`",
        f"- Models: `{', '.join(model_files)}`",
        f"- OOF: `{oof_name}`",
        "",
    ]
    (reports_dir / report_name).write_text("\n".join(lines), encoding="utf-8")

    print(f"metrics: {outputs_dir / metrics_name}")
    print(f"report: {reports_dir / report_name}")
    print(f"mean_rmsle: {mean_rmsle:.6f}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = root / "data" / INPUT_NAME
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")

    df = prepare_data(input_path)

    xgb_oof, xgb_folds, xgb_mean, xgb_models = run_cv(df, "xgb", build_xgb)
    write_outputs(df, xgb_oof, xgb_folds, xgb_mean, xgb_models, "xgb")

    rf_oof, rf_folds, rf_mean, rf_models = run_cv(df, "rf", build_rf)
    write_outputs(df, rf_oof, rf_folds, rf_mean, rf_models, "rf")


if __name__ == "__main__":
    main()
