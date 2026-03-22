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
SEEDS = [42, 2024, 3407]
N_TRIALS = 120


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
        "n_estimators": trial.suggest_int("n_estimators", 500, 3500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
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


def prepare_fold_data(df: pd.DataFrame, features: list[str]) -> tuple[list[int], dict]:
    folds = sorted(df[FOLD_COL].dropna().astype(int).unique().tolist())
    if len(folds) != 2:
        raise ValueError(f"Expect 2 folds, found: {folds}")

    fold_data = {}
    for fold in folds:
        tr = df[df[FOLD_COL] != fold].copy()
        va = df[df[FOLD_COL] == fold].copy()
        fill_values = tr[features].median(numeric_only=True).fillna(0)
        x_tr = tr[features].fillna(fill_values)
        x_va = va[features].fillna(fill_values)
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
    return folds, fold_data


def evaluate_params(folds: list[int], fold_data: dict, params: dict) -> tuple[list[FoldMetric], np.ndarray]:
    n_rows = 0
    for fd in fold_data.values():
        n_rows = max(n_rows, int(max(fd["valid_idx"])) + 1)
    oof_pred = np.zeros(n_rows, dtype=float)
    metrics: list[FoldMetric] = []

    for fold in folds:
        fd = fold_data[fold]
        model = XGBRegressor(**params)
        model.fit(fd["x_tr"], fd["y_tr"], eval_set=[(fd["x_va"], fd["y_va_log"])], verbose=False)
        pred = np.expm1(model.predict(fd["x_va"]))
        pred = np.clip(pred, 0, None)
        oof_pred[fd["valid_idx"]] = pred
        metrics.append(
            FoldMetric(
                fold=int(fold),
                train_size=int(fd["train_size"]),
                valid_size=int(fd["valid_size"]),
                rmsle=rmsle(fd["y_va_raw"], pred),
            )
        )
    return metrics, oof_pred


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

    df, added_feats = add_time_features(df)
    features = BASE_FEATURES + added_feats
    folds, fold_data = prepare_fold_data(df, features)

    candidates = []

    for seed in SEEDS:
        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial)
            scores = []
            for fold in folds:
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

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        best_params = suggest_params(optuna.trial.FixedTrial(study.best_params))
        fold_metrics, _ = evaluate_params(folds, fold_data, best_params)
        mean_r = float(np.mean([m.rmsle for m in fold_metrics]))
        std_r = float(np.std([m.rmsle for m in fold_metrics]))

        trials_df = study.trials_dataframe(
            attrs=("number", "value", "state", "params", "datetime_start", "datetime_complete")
        )
        trials_path = outputs_dir / f"xgboost_basic9_time_enhanced_bayes_multiseed_seed{seed}_tuning_history.csv"
        trials_df.to_csv(trials_path, index=False)

        best_path = outputs_dir / f"xgboost_basic9_time_enhanced_bayes_multiseed_seed{seed}_best_params.json"
        best_path.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "seed": seed,
                    "n_trials": N_TRIALS,
                    "study_best_value": float(study.best_value),
                    "study_best_params": study.best_params,
                    "resolved_best_params": best_params,
                    "fold_metrics": [asdict(m) for m in fold_metrics],
                    "mean_rmsle": mean_r,
                    "std_rmsle": std_r,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        candidates.append(
            {
                "seed": seed,
                "n_trials": N_TRIALS,
                "study_best_value": float(study.best_value),
                "best_params": best_params,
                "fold_metrics": [asdict(m) for m in fold_metrics],
                "mean_rmsle": mean_r,
                "std_rmsle": std_r,
                "tuning_history_path": trials_path.name,
                "best_params_path": best_path.name,
            }
        )

    # Select "most stable best": prioritize mean RMSLE, then std RMSLE.
    candidates_sorted = sorted(candidates, key=lambda x: (x["mean_rmsle"], x["std_rmsle"]))
    winner = candidates_sorted[0]

    winner_params = winner["best_params"]
    winner_fold_metrics, winner_oof = evaluate_params(folds, fold_data, winner_params)

    for fold in folds:
        fd = fold_data[fold]
        model = XGBRegressor(**winner_params)
        model.fit(fd["x_tr"], fd["y_tr"], eval_set=[(fd["x_va"], fd["y_va_log"])], verbose=False)
        model.save_model(str(models_dir / f"xgboost_basic9_time_enhanced_bayes_multiseed_2fold_seed{winner['seed']}_fold{fold}.json"))

    full_fill = df[features].median(numeric_only=True).fillna(0)
    x_full = df[features].fillna(full_fill)
    y_full = np.log1p(np.clip(df[TARGET].to_numpy(dtype=float), 0, None))
    full_model = XGBRegressor(**winner_params)
    full_model.fit(x_full, y_full, verbose=False)
    full_model_path = models_dir / "xgboost_basic9_time_enhanced_bayes_multiseed_2fold_full.json"
    full_model.save_model(str(full_model_path))

    oof = df[["datetime", "source", TARGET, FOLD_COL]].copy()
    oof["pred_count"] = winner_oof
    oof_path = outputs_dir / "oof_xgboost_basic9_time_enhanced_bayes_multiseed_2fold.csv"
    oof.to_csv(oof_path, index=False)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_data": data_path.name,
        "feature_strategy": "base9 + datetime_derived_time_features + bayesian_hpo_multiseed",
        "base_features": BASE_FEATURES,
        "added_time_features": added_feats,
        "features": features,
        "target": TARGET,
        "fold_column": FOLD_COL,
        "seeds": SEEDS,
        "n_trials_per_seed": N_TRIALS,
        "selection_rule": "sort by (mean_rmsle asc, std_rmsle asc)",
        "seed_results": candidates_sorted,
        "selected_seed": winner["seed"],
        "best_params": winner_params,
        "fold_metrics": [asdict(m) for m in winner_fold_metrics],
        "mean_rmsle": float(np.mean([m.rmsle for m in winner_fold_metrics])),
        "std_rmsle": float(np.std([m.rmsle for m in winner_fold_metrics])),
        "model_paths": [
            f"xgboost_basic9_time_enhanced_bayes_multiseed_2fold_seed{winner['seed']}_fold{f}.json"
            for f in folds
        ] + [full_model_path.name],
        "oof_path": oof_path.name,
    }
    metrics_path = outputs_dir / "xgboost_basic9_time_enhanced_bayes_multiseed_2fold_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = reports_dir / "xgboost_basic9_time_enhanced_bayes_multiseed_2fold_report.md"
    seed_summary_lines = []
    for c in candidates_sorted:
        fold_scores = ", ".join(f"{m['rmsle']:.6f}" for m in c["fold_metrics"])
        seed_summary_lines.append(
            f"- seed={c['seed']} | mean={c['mean_rmsle']:.6f} | std={c['std_rmsle']:.6f} | folds={fold_scores}"
        )
    report_path.write_text(
        "\n".join(
            [
                "# XGBoost Basic9 Time-Enhanced Bayesian Multi-Seed 2-Fold Report",
                "",
                f"- Time: {payload['timestamp']}",
                f"- Input: `{payload['input_data']}`",
                f"- Feature strategy: `{payload['feature_strategy']}`",
                f"- Seeds: `{SEEDS}`",
                f"- Trials per seed: `{N_TRIALS}`",
                f"- Selection rule: `{payload['selection_rule']}`",
                "",
                "## Seed Results",
                *seed_summary_lines,
                "",
                f"- Selected seed: `{payload['selected_seed']}`",
                f"- Final fold scores (RMSLE): `{', '.join(f'{m.rmsle:.6f}' for m in winner_fold_metrics)}`",
                f"- Final mean RMSLE: `{payload['mean_rmsle']:.6f}`",
                f"- Final std RMSLE: `{payload['std_rmsle']:.6f}`",
                f"- Models: `{', '.join(payload['model_paths'])}`",
                f"- OOF: `{payload['oof_path']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    print(f"selected_seed: {payload['selected_seed']}")
    print(f"mean_rmsle: {payload['mean_rmsle']:.6f}")
    print(f"std_rmsle: {payload['std_rmsle']:.6f}")


if __name__ == "__main__":
    main()
