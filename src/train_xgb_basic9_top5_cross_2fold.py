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


def get_top5_by_gain(df: pd.DataFrame) -> list[str]:
    x = df[BASE_FEATURES].copy()
    fill_values = x.median(numeric_only=True).fillna(0)
    x = x.fillna(fill_values)
    y = np.log1p(np.clip(df[TARGET].to_numpy(dtype=float), 0, None))

    model = build_model()
    model.fit(x, y, verbose=False)
    score = model.get_booster().get_score(importance_type="gain")

    gain = {f: float(score.get(f, 0.0)) for f in BASE_FEATURES}
    ranked = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
    return [f for f, _ in ranked[:5]]


def make_cross_features(df: pd.DataFrame, top5: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    cross_cols: list[str] = []
    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            a, b = top5[i], top5[j]
            c = f"cross_{a}_x_{b}"
            out[c] = out[a] * out[b]
            cross_cols.append(c)
    return out, cross_cols


def cv_eval(df: pd.DataFrame, features: list[str]) -> tuple[list[FoldMetric], np.ndarray]:
    unique_folds = sorted(df[FOLD_COL].dropna().astype(int).unique().tolist())
    if len(unique_folds) != 2:
        raise ValueError(f"Expect 2 folds, found: {unique_folds}")

    oof = np.zeros(len(df), dtype=float)
    metrics: list[FoldMetric] = []
    for fold in unique_folds:
        tr_idx = df.index[df[FOLD_COL] != fold]
        va_idx = df.index[df[FOLD_COL] == fold]
        tr = df.loc[tr_idx].copy()
        va = df.loc[va_idx].copy()

        fill_values = tr[features].median(numeric_only=True).fillna(0)
        x_tr = tr[features].fillna(fill_values)
        x_va = va[features].fillna(fill_values)
        y_tr = np.log1p(np.clip(tr[TARGET].to_numpy(dtype=float), 0, None))
        y_va = va[TARGET].to_numpy(dtype=float)

        model = build_model()
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


def train_and_save_full(df: pd.DataFrame, features: list[str], model_path: Path) -> None:
    x = df[features].copy()
    fill_values = x.median(numeric_only=True).fillna(0)
    x = x.fillna(fill_values)
    y = np.log1p(np.clip(df[TARGET].to_numpy(dtype=float), 0, None))
    model = build_model()
    model.fit(x, y, verbose=False)
    model.save_model(str(model_path))


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
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    top5 = get_top5_by_gain(df)
    cross_df, cross_cols = make_cross_features(df, top5)
    cross_features = BASE_FEATURES + cross_cols

    base_fold_metrics, base_oof = cv_eval(cross_df, BASE_FEATURES)
    cross_fold_metrics, cross_oof = cv_eval(cross_df, cross_features)
    base_mean = float(np.mean([m.rmsle for m in base_fold_metrics]))
    cross_mean = float(np.mean([m.rmsle for m in cross_fold_metrics]))

    # Save full model with cross features
    full_model_path = models_dir / "xgboost_basic9_top5cross_2fold_full.json"
    train_and_save_full(cross_df, cross_features, full_model_path)

    oof = cross_df[["datetime", "source", TARGET, FOLD_COL]].copy()
    oof["pred_base9"] = base_oof
    oof["pred_base9_top5cross"] = cross_oof
    oof_path = outputs_dir / "oof_xgboost_basic9_top5cross_2fold.csv"
    oof.to_csv(oof_path, index=False)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_data": data_path.name,
        "base_features": BASE_FEATURES,
        "top5_by_gain": top5,
        "cross_features": cross_cols,
        "n_cross_features": len(cross_cols),
        "baseline_fold_metrics": [asdict(m) for m in base_fold_metrics],
        "baseline_mean_rmsle": base_mean,
        "cross_fold_metrics": [asdict(m) for m in cross_fold_metrics],
        "cross_mean_rmsle": cross_mean,
        "delta_rmsle": cross_mean - base_mean,
        "model_paths": [full_model_path.name],
        "oof_path": oof_path.name,
    }
    metrics_path = outputs_dir / "xgboost_basic9_top5cross_2fold_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = reports_dir / "xgboost_basic9_top5cross_2fold_report.md"
    report_lines = [
        "# XGBoost Basic9 + Top5 Cross 2-Fold Report",
        "",
        f"- Time: {payload['timestamp']}",
        f"- Input: `{payload['input_data']}`",
        f"- Top5 features by gain: `{', '.join(top5)}`",
        f"- Added cross features: `{len(cross_cols)}`",
        f"- Baseline fold RMSLE: `{', '.join(f'{m.rmsle:.6f}' for m in base_fold_metrics)}`",
        f"- Baseline mean RMSLE: `{base_mean:.6f}`",
        f"- Cross fold RMSLE: `{', '.join(f'{m.rmsle:.6f}' for m in cross_fold_metrics)}`",
        f"- Cross mean RMSLE: `{cross_mean:.6f}`",
        f"- Delta (cross - baseline): `{(cross_mean - base_mean):.6f}`",
        f"- Full model: `{full_model_path.name}`",
        f"- OOF: `{oof_path.name}`",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    print(f"baseline_mean_rmsle: {base_mean:.6f}")
    print(f"cross_mean_rmsle: {cross_mean:.6f}")
    print(f"top5_by_gain: {', '.join(top5)}")


if __name__ == "__main__":
    main()
