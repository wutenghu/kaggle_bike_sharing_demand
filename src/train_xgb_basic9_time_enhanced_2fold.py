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


def build_model(best_params: dict | None = None) -> XGBRegressor:
    if best_params:
        return XGBRegressor(**best_params)
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


def add_time_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["month"] = out["datetime"].dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_peak_hour"] = out["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["dayofweek"] / 7.0)
    out["month_sin"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["month"] / 12.0)

    added = [
        "dayofweek",
        "month",
        "is_weekend",
        "is_peak_hour",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]
    return out, added


def cv_eval(df: pd.DataFrame, features: list[str], model_params: dict | None = None) -> tuple[list[FoldMetric], np.ndarray]:
    folds = sorted(df[FOLD_COL].dropna().astype(int).unique().tolist())
    if len(folds) != 2:
        raise ValueError(f"Expect 2 folds, found: {folds}")

    oof = np.zeros(len(df), dtype=float)
    metrics: list[FoldMetric] = []
    for fold in folds:
        tr_idx = df.index[df[FOLD_COL] != fold]
        va_idx = df.index[df[FOLD_COL] == fold]
        tr = df.loc[tr_idx].copy()
        va = df.loc[va_idx].copy()

        fill_values = tr[features].median(numeric_only=True).fillna(0)
        x_tr = tr[features].fillna(fill_values)
        x_va = va[features].fillna(fill_values)
        y_tr = np.log1p(np.clip(tr[TARGET].to_numpy(dtype=float), 0, None))
        y_va = va[TARGET].to_numpy(dtype=float)

        model = build_model(model_params)
        model.fit(x_tr, y_tr, verbose=False)
        pred = np.expm1(model.predict(x_va))
        pred = np.clip(pred, 0, None)
        oof[va_idx] = pred

        metrics.append(
            FoldMetric(
                fold=int(fold),
                train_size=int(len(tr)),
                valid_size=int(len(va)),
                rmsle=rmsle(y_va, pred),
            )
        )

    return metrics, oof


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "cv2_random_train_miss_train_20260322_205429.csv"
    outputs_dir = root / "outputs"
    models_dir = root / "models"
    reports_dir = root / "reports"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Reuse tuned basic9 params to isolate feature-effect.
    tuned_ref = outputs_dir / "xgboost_basic9_bayes_2fold_metrics.json"
    tuned_params = None
    if tuned_ref.exists():
        tuned_params = json.loads(tuned_ref.read_text(encoding="utf-8")).get("best_params")

    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["hour"] = df["datetime"].dt.hour

    for c in BASE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    enhanced_df, added_feats = add_time_features(df)
    enhanced_features = BASE_FEATURES + added_feats

    base_metrics, base_oof = cv_eval(enhanced_df, BASE_FEATURES, tuned_params)
    enh_metrics, enh_oof = cv_eval(enhanced_df, enhanced_features, tuned_params)
    base_mean = float(np.mean([m.rmsle for m in base_metrics]))
    enh_mean = float(np.mean([m.rmsle for m in enh_metrics]))

    # Save full enhanced model
    fill_values = enhanced_df[enhanced_features].median(numeric_only=True).fillna(0)
    x_full = enhanced_df[enhanced_features].fillna(fill_values)
    y_full = np.log1p(np.clip(enhanced_df[TARGET].to_numpy(dtype=float), 0, None))
    full_model = build_model(tuned_params)
    full_model.fit(x_full, y_full, verbose=False)
    full_model_path = models_dir / "xgboost_basic9_time_enhanced_2fold_full.json"
    full_model.save_model(str(full_model_path))

    oof = enhanced_df[["datetime", "source", TARGET, FOLD_COL]].copy()
    oof["pred_base9"] = base_oof
    oof["pred_time_enhanced"] = enh_oof
    oof_path = outputs_dir / "oof_xgboost_basic9_time_enhanced_2fold.csv"
    oof.to_csv(oof_path, index=False)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_data": data_path.name,
        "base_features": BASE_FEATURES,
        "added_time_features": added_feats,
        "baseline_fold_metrics": [asdict(m) for m in base_metrics],
        "baseline_mean_rmsle": base_mean,
        "enhanced_fold_metrics": [asdict(m) for m in enh_metrics],
        "enhanced_mean_rmsle": enh_mean,
        "delta_rmsle": enh_mean - base_mean,
        "model_paths": [full_model_path.name],
        "oof_path": oof_path.name,
        "using_params_from": tuned_ref.name if tuned_ref.exists() else None,
    }
    metrics_path = outputs_dir / "xgboost_basic9_time_enhanced_2fold_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = reports_dir / "xgboost_basic9_time_enhanced_2fold_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# XGBoost Basic9 Time-Enhanced 2-Fold Report",
                "",
                f"- Time: {payload['timestamp']}",
                f"- Input: `{payload['input_data']}`",
                f"- Added time features: `{', '.join(added_feats)}`",
                f"- Baseline fold RMSLE: `{', '.join(f'{m.rmsle:.6f}' for m in base_metrics)}`",
                f"- Baseline mean RMSLE: `{base_mean:.6f}`",
                f"- Enhanced fold RMSLE: `{', '.join(f'{m.rmsle:.6f}' for m in enh_metrics)}`",
                f"- Enhanced mean RMSLE: `{enh_mean:.6f}`",
                f"- Delta (enhanced - baseline): `{(enh_mean - base_mean):.6f}`",
                f"- Full model: `{full_model_path.name}`",
                f"- OOF: `{oof_path.name}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    print(f"baseline_mean_rmsle: {base_mean:.6f}")
    print(f"enhanced_mean_rmsle: {enh_mean:.6f}")


if __name__ == "__main__":
    main()
