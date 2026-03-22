# XGBoost Basic9 HistStats TimePrefix 2-Fold Calibration Report

- Time: 2026-03-23 00:31:28
- Input OOF: `oof_xgboost_basic9_histstats_timeprefix_2fold.csv`
- Baseline RMSLE: `0.307747`
- Month log-shift RMSLE: `0.307516` (delta `-0.000232`)
- Year-Month log-shift RMSLE: `0.307366` (delta `-0.000381`)
- Best strategy: `year_month_logshift`
- Calibrated OOF: `oof_xgboost_basic9_histstats_timeprefix_2fold_calibrated.csv`
- Figure: `oof_daily_raw_vs_calibrated_2fold.png`

## Bias by Year (mean_log_bias, pred_true_ratio)

- baseline: 2011: log_bias=0.003310, ratio=0.978037, rmsle=0.324844; 2012: log_bias=-0.000735, ratio=0.972729, rmsle=0.289643
- month_logshift: 2011: log_bias=0.002022, ratio=0.977798, rmsle=0.324393; 2012: log_bias=-0.002022, ratio=0.972070, rmsle=0.289657
- year_month_logshift: 2011: log_bias=0.000000, ratio=0.975581, rmsle=0.324252; 2012: log_bias=0.000000, ratio=0.974304, rmsle=0.289498
