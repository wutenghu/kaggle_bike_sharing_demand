from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
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
TOP_K = 30


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


def get_tsfresh_topk_by_shap(
    df: pd.DataFrame,
    model_path: Path,
    candidate_features: list[str],
    top_k: int = TOP_K,
) -> list[str]:
    x = df[candidate_features].copy()
    for c in candidate_features:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    fill_values = x.median(numeric_only=True)
    x = x.fillna(fill_values)

    # Sample for stable and faster SHAP ranking
    if len(x) > 4000:
        x = x.sample(n=4000, random_state=42)

    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    booster = model.get_booster()

    dm = xgb.DMatrix(x, feature_names=candidate_features)
    contrib = booster.predict(dm, pred_contribs=True)
    shap_values = contrib[:, :-1]

    importance = np.mean(np.abs(shap_values), axis=0)
    imp_df = pd.DataFrame({"feature": candidate_features, "mean_abs_shap": importance})

    tsfresh_only = imp_df[imp_df["feature"].str.startswith("count__")].copy()
    tsfresh_only = tsfresh_only.sort_values("mean_abs_shap", ascending=False)
    return tsfresh_only.head(top_k)["feature"].tolist()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    outputs_dir = root / "outputs"
    models_dir = root / "models"
    reports_dir = root / "reports"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ref_metrics_path = outputs_dir / "xgboost_tsfresh_2fold_metrics.json"
    if not ref_metrics_path.exists():
        raise FileNotFoundError("Missing outputs/xgboost_tsfresh_2fold_metrics.json")
    ref = json.loads(ref_metrics_path.read_text(encoding="utf-8"))

    input_path = data_dir / ref["input_data"]
    ref_model_path = models_dir / "xgboost_tsfresh_2fold_full.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input data: {input_path}")
    if not ref_model_path.exists():
        raise FileNotFoundError(f"Missing reference model: {ref_model_path}")

    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["hour"] = df["datetime"].dt.hour
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    for c in BASE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    tsfresh_candidates = ref.get("full_selected_tsfresh_features", [])
    features_for_shap = BASE_FEATURES + tsfresh_candidates
    for c in tsfresh_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    train_df = df.dropna(subset=[TARGET]).copy().reset_index(drop=True)

    top30_tsfresh = get_tsfresh_topk_by_shap(
        train_df, ref_model_path, features_for_shap, top_k=TOP_K
    )
    final_features = BASE_FEATURES + top30_tsfresh

    oof_pred = np.zeros(len(train_df), dtype=float)
    fold_metrics: list[FoldMetric] = []

    unique_folds = sorted(train_df[FOLD_COL].dropna().astype(int).unique().tolist())
    if len(unique_folds) != 2:
        raise ValueError(f"Expect 2 folds, found: {unique_folds}")

    for fold in unique_folds:
        tr_idx = train_df.index[train_df[FOLD_COL] != fold]
        va_idx = train_df.index[train_df[FOLD_COL] == fold]

        tr = train_df.loc[tr_idx].copy()
        va = train_df.loc[va_idx].copy()

        fill_values = tr[final_features].median(numeric_only=True)
        x_tr = tr[final_features].fillna(fill_values)
        x_va = va[final_features].fillna(fill_values)
        y_tr = np.log1p(np.clip(tr[TARGET].to_numpy(dtype=float), 0, None))
        y_va = va[TARGET].to_numpy(dtype=float)

        model = build_model()
        model.fit(x_tr, y_tr, eval_set=[(x_va, np.log1p(np.clip(y_va, 0, None)))], verbose=False)

        pred = np.expm1(model.predict(x_va))
        pred = np.clip(pred, 0, None)
        oof_pred[va_idx] = pred

        fold_score = rmsle(y_va, pred)
        fold_metrics.append(
            FoldMetric(
                fold=int(fold),
                train_size=int(len(tr)),
                valid_size=int(len(va)),
                rmsle=fold_score,
            )
        )

        model.save_model(str(models_dir / f"xgboost_tsfresh_top30_2fold_fold{fold}.json"))

    mean_rmsle = float(np.mean([m.rmsle for m in fold_metrics]))

    full_fill = train_df[final_features].median(numeric_only=True)
    full_model = build_model()
    full_model.fit(
        train_df[final_features].fillna(full_fill),
        np.log1p(np.clip(train_df[TARGET].to_numpy(dtype=float), 0, None)),
        verbose=False,
    )
    full_model_path = models_dir / "xgboost_tsfresh_top30_2fold_full.json"
    full_model.save_model(str(full_model_path))

    oof = train_df[["datetime", "source", TARGET, FOLD_COL]].copy()
    oof["pred_count"] = oof_pred
    oof_path = outputs_dir / "oof_xgboost_tsfresh_top30_2fold.csv"
    oof.to_csv(oof_path, index=False)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_data": input_path.name,
        "feature_strategy": "base_9_features + tsfresh_shap_top30",
        "base_features": BASE_FEATURES,
        "tsfresh_top30_features": top30_tsfresh,
        "target": TARGET,
        "fold_column": FOLD_COL,
        "fold_metrics": [asdict(m) for m in fold_metrics],
        "mean_rmsle": mean_rmsle,
        "model_paths": [
            "xgboost_tsfresh_top30_2fold_fold0.json",
            "xgboost_tsfresh_top30_2fold_fold1.json",
            full_model_path.name,
        ],
        "oof_path": oof_path.name,
    }
    metrics_path = outputs_dir / "xgboost_tsfresh_top30_2fold_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = reports_dir / "xgboost_tsfresh_top30_2fold_report.md"
    fold_scores_str = ", ".join(f"{m.rmsle:.6f}" for m in fold_metrics)
    report_path.write_text(
        "\n".join(
            [
                "# XGBoost TSFresh Top30 2-Fold CV Report",
                "",
                f"- Time: {payload['timestamp']}",
                f"- Input: `{payload['input_data']}`",
                f"- Feature strategy: `{payload['feature_strategy']}`",
                f"- Fold scores (RMSLE): `{fold_scores_str}`",
                f"- Mean RMSLE: `{mean_rmsle:.6f}`",
                f"- Models: `{', '.join(payload['model_paths'])}`",
                f"- OOF: `{payload['oof_path']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    print(f"mean_rmsle: {mean_rmsle:.6f}")


if __name__ == "__main__":
    main()
