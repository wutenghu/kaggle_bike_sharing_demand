from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET = "count"
PRED_COL = "pred_count"


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def bias_by_year(df: pd.DataFrame, pred_col: str) -> list[dict]:
    rows = []
    for year, g in df.groupby(df["datetime"].dt.year):
        y = g[TARGET].to_numpy(dtype=float)
        p = g[pred_col].to_numpy(dtype=float)
        rows.append(
            {
                "year": int(year),
                "n": int(len(g)),
                "true_mean": float(np.mean(y)),
                "pred_mean": float(np.mean(p)),
                "pred_true_ratio": float(np.mean(p) / np.mean(y)) if np.mean(y) != 0 else None,
                "mean_log_bias": float(np.mean(np.log1p(np.clip(p, 0, None)) - np.log1p(np.clip(y, 0, None)))),
                "rmsle": rmsle(y, p),
            }
        )
    return rows


def apply_log_shift_calibration(
    df: pd.DataFrame,
    group_key: str,
    src_pred_col: str,
    out_col: str,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    log_y = np.log1p(np.clip(out[TARGET].to_numpy(dtype=float), 0, None))
    log_p = np.log1p(np.clip(out[src_pred_col].to_numpy(dtype=float), 0, None))
    out["_resid_log"] = log_y - log_p

    if group_key == "month":
        grp = out.groupby(out["datetime"].dt.month)
        key_vals = out["datetime"].dt.month
    elif group_key == "year_month":
        ym = out["datetime"].dt.strftime("%Y-%m")
        grp = out.groupby(ym)
        key_vals = ym
    else:
        raise ValueError(group_key)

    shift_map = grp["_resid_log"].mean().to_dict()
    global_shift = float(out["_resid_log"].mean())

    shifts = key_vals.map(shift_map).fillna(global_shift).to_numpy(dtype=float)
    out[out_col] = np.expm1(log_p + shifts)
    out[out_col] = np.clip(out[out_col].to_numpy(dtype=float), 0, None)

    return out.drop(columns=["_resid_log"]), {
        "group_key": group_key,
        "n_groups": int(len(shift_map)),
        "global_shift": global_shift,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    oof_path = root / "outputs" / "oof_xgboost_basic9_histstats_timeprefix_2fold.csv"
    outputs_dir = root / "outputs"
    reports_dir = root / "reports"
    figures_dir = root / "figures"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(oof_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df[PRED_COL] = pd.to_numeric(df[PRED_COL], errors="coerce")
    df = df.dropna(subset=["datetime", TARGET, PRED_COL]).copy()

    df["pred_raw"] = np.clip(df[PRED_COL].to_numpy(dtype=float), 0, None)

    base_rmsle = rmsle(df[TARGET].to_numpy(dtype=float), df["pred_raw"].to_numpy(dtype=float))
    base_bias = bias_by_year(df, "pred_raw")

    month_df, month_meta = apply_log_shift_calibration(df, "month", "pred_raw", "pred_cal_month_logshift")
    ym_df, ym_meta = apply_log_shift_calibration(df, "year_month", "pred_raw", "pred_cal_ym_logshift")

    month_rmsle = rmsle(month_df[TARGET].to_numpy(dtype=float), month_df["pred_cal_month_logshift"].to_numpy(dtype=float))
    ym_rmsle = rmsle(ym_df[TARGET].to_numpy(dtype=float), ym_df["pred_cal_ym_logshift"].to_numpy(dtype=float))

    month_bias = bias_by_year(month_df, "pred_cal_month_logshift")
    ym_bias = bias_by_year(ym_df, "pred_cal_ym_logshift")

    merged = df[["datetime", "source", TARGET, "cv_fold", "pred_raw"]].copy()
    merged = merged.merge(
        month_df[["datetime", "pred_cal_month_logshift"]],
        on="datetime",
        how="left",
    )
    merged = merged.merge(
        ym_df[["datetime", "pred_cal_ym_logshift"]],
        on="datetime",
        how="left",
    )

    oof_cal_path = outputs_dir / "oof_xgboost_basic9_histstats_timeprefix_2fold_calibrated.csv"
    merged.to_csv(oof_cal_path, index=False)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_oof": oof_path.name,
        "baseline": {
            "rmsle": base_rmsle,
            "bias_by_year": base_bias,
        },
        "month_logshift": {
            **month_meta,
            "rmsle": month_rmsle,
            "delta_vs_baseline": month_rmsle - base_rmsle,
            "bias_by_year": month_bias,
        },
        "year_month_logshift": {
            **ym_meta,
            "rmsle": ym_rmsle,
            "delta_vs_baseline": ym_rmsle - base_rmsle,
            "bias_by_year": ym_bias,
        },
        "best_strategy": "year_month_logshift" if ym_rmsle <= month_rmsle else "month_logshift",
        "calibrated_oof_path": oof_cal_path.name,
    }

    metrics_path = outputs_dir / "xgboost_basic9_histstats_timeprefix_2fold_calibration_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plot daily aggregates for raw vs calibrated(best)
    best_col = "pred_cal_ym_logshift" if payload["best_strategy"] == "year_month_logshift" else "pred_cal_month_logshift"
    daily = merged.copy()
    daily["date"] = daily["datetime"].dt.floor("D")
    d = daily.groupby("date", as_index=False)[[TARGET, "pred_raw", best_col]].sum()

    fig_path = figures_dir / "oof_daily_raw_vs_calibrated_2fold.png"
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(d["date"], d[TARGET], color="#1f77b4", linewidth=1.8, label="actual")
    ax.plot(d["date"], d["pred_raw"], color="#ff7f0e", linewidth=1.5, label="pred_raw")
    ax.plot(d["date"], d[best_col], color="#2ca02c", linewidth=1.5, label=f"pred_calibrated ({payload['best_strategy']})")
    ax.set_title("OOF Daily Count: Raw vs Calibrated")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count per Day")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    report_path = reports_dir / "xgboost_basic9_histstats_timeprefix_2fold_calibration_report.md"
    report_lines = [
        "# XGBoost Basic9 HistStats TimePrefix 2-Fold Calibration Report",
        "",
        f"- Time: {payload['timestamp']}",
        f"- Input OOF: `{payload['input_oof']}`",
        f"- Baseline RMSLE: `{base_rmsle:.6f}`",
        f"- Month log-shift RMSLE: `{month_rmsle:.6f}` (delta `{month_rmsle - base_rmsle:.6f}`)",
        f"- Year-Month log-shift RMSLE: `{ym_rmsle:.6f}` (delta `{ym_rmsle - base_rmsle:.6f}`)",
        f"- Best strategy: `{payload['best_strategy']}`",
        f"- Calibrated OOF: `{oof_cal_path.name}`",
        f"- Figure: `{fig_path.name}`",
        "",
        "## Bias by Year (mean_log_bias, pred_true_ratio)",
        "",
    ]

    def fmt_bias(rows: list[dict]) -> str:
        return "; ".join(
            f"{r['year']}: log_bias={r['mean_log_bias']:.6f}, ratio={r['pred_true_ratio']:.6f}, rmsle={r['rmsle']:.6f}"
            for r in rows
        )

    report_lines.append(f"- baseline: {fmt_bias(base_bias)}")
    report_lines.append(f"- month_logshift: {fmt_bias(month_bias)}")
    report_lines.append(f"- year_month_logshift: {fmt_bias(ym_bias)}")
    report_lines.append("")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"metrics={metrics_path}")
    print(f"report={report_path}")
    print(f"figure={fig_path}")
    print(f"baseline_rmsle={base_rmsle:.6f}")
    print(f"month_logshift_rmsle={month_rmsle:.6f}")
    print(f"year_month_logshift_rmsle={ym_rmsle:.6f}")
    print(f"best_strategy={payload['best_strategy']}")


if __name__ == "__main__":
    main()
