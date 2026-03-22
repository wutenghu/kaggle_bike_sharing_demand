from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
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
N_TRIALS = 40


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


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        "random_state": 42,
        "n_jobs": 4,
        "tree_method": "hist",
    }


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
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    for c in BASE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    unique_folds = sorted(df[FOLD_COL].dropna().astype(int).unique().tolist())
    if len(unique_folds) != 2:
        raise ValueError(f"Expect 2 folds, found: {unique_folds}")

    fold_data = {}
    for fold in unique_folds:
        tr = df[df[FOLD_COL] != fold].copy()
        va = df[df[FOLD_COL] == fold].copy()
        fill_values = tr[BASE_FEATURES].median(numeric_only=True).fillna(0)
        x_tr = tr[BASE_FEATURES].fillna(fill_values)
        x_va = va[BASE_FEATURES].fillna(fill_values)
        y_tr = np.log1p(np.clip(tr[TARGET].to_numpy(dtype=float), 0, None))
        y_va_raw = va[TARGET].to_numpy(dtype=float)
        y_va_log = np.log1p(np.clip(y_va_raw, 0, None))
        fold_data[fold] = {
            "train_idx": tr.index.to_numpy(),
            "valid_idx": va.index.to_numpy(),
            "x_tr": x_tr,
            "x_va": x_va,
            "y_tr": y_tr,
            "y_va_raw": y_va_raw,
            "y_va_log": y_va_log,
            "train_size": len(tr),
            "valid_size": len(va),
        }

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        scores = []
        for fold in unique_folds:
            fd = fold_data[fold]
            model = XGBRegressor(**params)
            model.fit(fd["x_tr"], fd["y_tr"], eval_set=[(fd["x_va"], fd["y_va_log"])], verbose=False)
            pred = np.expm1(model.predict(fd["x_va"]))
            score = rmsle(fd["y_va_raw"], pred)
            scores.append(score)
            trial.report(float(np.mean(scores)), step=int(fold))
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = suggest_params(optuna.trial.FixedTrial(study.best_params))
    best_value = float(study.best_value)

    oof_pred = np.zeros(len(df), dtype=float)
    fold_metrics: list[FoldMetric] = []
    for fold in unique_folds:
        fd = fold_data[fold]
        model = XGBRegressor(**best_params)
        model.fit(fd["x_tr"], fd["y_tr"], eval_set=[(fd["x_va"], fd["y_va_log"])], verbose=False)
        pred = np.expm1(model.predict(fd["x_va"]))
        pred = np.clip(pred, 0, None)
        oof_pred[fd["valid_idx"]] = pred
        fold_metrics.append(
            FoldMetric(
                fold=int(fold),
                train_size=int(fd["train_size"]),
                valid_size=int(fd["valid_size"]),
                rmsle=rmsle(fd["y_va_raw"], pred),
            )
        )
        model.save_model(str(models_dir / f"xgboost_basic9_bayes_2fold_fold{fold}.json"))

    full_fill = df[BASE_FEATURES].median(numeric_only=True).fillna(0)
    x_full = df[BASE_FEATURES].fillna(full_fill)
    y_full = np.log1p(np.clip(df[TARGET].to_numpy(dtype=float), 0, None))
    full_model = XGBRegressor(**best_params)
    full_model.fit(x_full, y_full, verbose=False)
    full_model_path = models_dir / "xgboost_basic9_bayes_2fold_full.json"
    full_model.save_model(str(full_model_path))

    oof = df[["datetime", "source", TARGET, FOLD_COL]].copy()
    oof["pred_count"] = oof_pred
    oof_path = outputs_dir / "oof_xgboost_basic9_bayes_2fold.csv"
    oof.to_csv(oof_path, index=False)

    trials_df = study.trials_dataframe(
        attrs=("number", "value", "state", "params", "datetime_start", "datetime_complete")
    )
    tuning_path = outputs_dir / "xgboost_basic9_bayes_2fold_tuning_history.csv"
    trials_df.to_csv(tuning_path, index=False)

    best_params_path = outputs_dir / "xgboost_basic9_bayes_2fold_best_params.json"
    best_params_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "study_best_value": best_value,
                "study_best_params": study.best_params,
                "resolved_best_params": best_params,
                "n_trials": N_TRIALS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    mean_rmsle = float(np.mean([m.rmsle for m in fold_metrics]))
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_data": data_path.name,
        "feature_strategy": "base9 + bayesian_hpo",
        "features": BASE_FEATURES,
        "target": TARGET,
        "fold_column": FOLD_COL,
        "n_trials": N_TRIALS,
        "study_best_value": best_value,
        "best_params": best_params,
        "fold_metrics": [asdict(m) for m in fold_metrics],
        "mean_rmsle": mean_rmsle,
        "model_paths": [
            "xgboost_basic9_bayes_2fold_fold0.json",
            "xgboost_basic9_bayes_2fold_fold1.json",
            full_model_path.name,
        ],
        "oof_path": oof_path.name,
        "tuning_history_path": tuning_path.name,
        "best_params_path": best_params_path.name,
    }
    metrics_path = outputs_dir / "xgboost_basic9_bayes_2fold_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = reports_dir / "xgboost_basic9_bayes_2fold_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# XGBoost Basic9 Bayesian 2-Fold Report",
                "",
                f"- Time: {payload['timestamp']}",
                f"- Input: `{payload['input_data']}`",
                f"- Feature strategy: `{payload['feature_strategy']}`",
                f"- Trials: `{N_TRIALS}`",
                f"- Fold scores (RMSLE): `{', '.join(f'{m.rmsle:.6f}' for m in fold_metrics)}`",
                f"- Mean RMSLE: `{mean_rmsle:.6f}`",
                f"- Study best value: `{best_value:.6f}`",
                f"- Models: `{', '.join(payload['model_paths'])}`",
                f"- OOF: `{payload['oof_path']}`",
                f"- Tuning history: `{payload['tuning_history_path']}`",
                f"- Best params: `{payload['best_params_path']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    print(f"mean_rmsle: {mean_rmsle:.6f}")
    print(f"study_best_value: {best_value:.6f}")


if __name__ == "__main__":
    main()
