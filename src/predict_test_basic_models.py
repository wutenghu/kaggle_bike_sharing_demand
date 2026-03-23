from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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


def infer_count(model, x: pd.DataFrame) -> np.ndarray:
    pred = np.expm1(model.predict(x))
    pred = np.clip(pred, 0, None)
    pred = np.rint(pred).astype(int)
    return pred


def write_submission(path: Path, template: pd.DataFrame, pred_df: pd.DataFrame, pred_col: str) -> None:
    merged = template[["datetime"]].merge(pred_df[["datetime", pred_col]], on="datetime", how="left")
    if merged[pred_col].isna().any():
        n_miss = int(merged[pred_col].isna().sum())
        raise ValueError(f"Missing predictions after alignment: {n_miss}")
    out = pd.DataFrame({
        "datetime": merged["datetime"],
        "count": merged[pred_col].astype(int),
    })
    out.to_csv(path, index=False)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    outputs_dir = root / "outputs"
    models_dir = root / "models"
    reports_dir = root / "reports"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    merged_path = data_dir / "train_test_merged.csv"
    template_path = data_dir / "sampleSubmission_expected.csv"
    xgb_model_path = models_dir / "xgboost_basic_9feat_hour_full.pkl"
    rf_model_path = models_dir / "randomforest_basic_9feat_hour_full.pkl"

    for p in [merged_path, template_path, xgb_model_path, rf_model_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    df = pd.read_csv(merged_path)
    template = pd.read_csv(template_path)

    test_df = df[df["source"].eq("test")].copy()
    test_df["datetime"] = pd.to_datetime(test_df["datetime"], errors="coerce")
    template["datetime"] = pd.to_datetime(template["datetime"], errors="coerce")

    for c in FEATURES:
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

    fill_values = test_df[FEATURES].median(numeric_only=True)
    x_test = test_df[FEATURES].fillna(fill_values)

    xgb_model = joblib.load(xgb_model_path)
    rf_model = joblib.load(rf_model_path)

    test_df["pred_xgb"] = infer_count(xgb_model, x_test)
    test_df["pred_rf"] = infer_count(rf_model, x_test)

    test_pred = test_df[["datetime", "pred_xgb", "pred_rf"]].copy()

    xgb_out = outputs_dir / "submission_xgboost_basic9_full.csv"
    rf_out = outputs_dir / "submission_randomforest_basic9_full.csv"

    write_submission(xgb_out, template, test_pred, "pred_xgb")
    write_submission(rf_out, template, test_pred, "pred_rf")

    # Write compact compare report using metrics + submission checks.
    xgb_metrics_path = outputs_dir / "xgboost_basic_2fold_metrics.json"
    rf_metrics_path = outputs_dir / "randomforest_basic_2fold_metrics.json"
    if not xgb_metrics_path.exists() or not rf_metrics_path.exists():
        raise FileNotFoundError("Missing metrics json for compare report")

    xgb_metrics = json.loads(xgb_metrics_path.read_text(encoding="utf-8"))
    rf_metrics = json.loads(rf_metrics_path.read_text(encoding="utf-8"))

    xgb_sub = pd.read_csv(xgb_out)
    rf_sub = pd.read_csv(rf_out)

    report_lines = [
        "# Basic9 Train+MissTrain Model Compare",
        "",
        f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Metric: RMSLE (lower is better)",
        "",
        "## CV Results",
        f"- XGBoost mean RMSLE: `{xgb_metrics['mean_rmsle']:.6f}`",
        f"- RandomForest mean RMSLE: `{rf_metrics['mean_rmsle']:.6f}`",
        "",
        "## Submission Artifacts",
        f"- `outputs/{xgb_out.name}` rows: `{len(xgb_sub)}`",
        f"- `outputs/{rf_out.name}` rows: `{len(rf_sub)}`",
        "- Alignment target: `data/sampleSubmission_expected.csv` rows: `6493`",
        "",
    ]

    report_path = reports_dir / "basic9_model_compare_train_miss_train.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"xgb_submission: {xgb_out}")
    print(f"rf_submission: {rf_out}")
    print(f"compare_report: {report_path}")


if __name__ == "__main__":
    main()
