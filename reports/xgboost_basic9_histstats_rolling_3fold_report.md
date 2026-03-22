# XGBoost Basic9 HistStats Rolling 3-Fold Report

- Time: 2026-03-23 00:08:35
- Input: `cv2_random_train_miss_train_20260322_205429.csv`
- Feature strategy: `base9 + leakage_safe_historical_mean_features`
- Added features: `hist_mean_global, hist_mean_hour, hist_mean_dayofweek, hist_mean_month, hist_mean_hour_x_workingday, hist_mean_hour_x_season`
- Hist rule: `train rows use past-only stats; valid rows map stats from train segment only`
- CV strategy: `time_rolling_expanding_window_3fold`
- Fold definition: `split sorted samples into 4 contiguous blocks; folds validate on block 2/3/4`
- Fold RMSLE: `0.447657, 0.435793, 0.456434`
- Mean RMSLE: `0.446628`
- Std RMSLE: `0.008458`
- Params source: `xgboost_basic9_bayes_2fold_metrics.json`
- Models: `xgboost_basic9_histstats_rolling_3fold_fold0.json, xgboost_basic9_histstats_rolling_3fold_fold1.json, xgboost_basic9_histstats_rolling_3fold_fold2.json, xgboost_basic9_histstats_rolling_3fold_full.json`
- OOF: `oof_xgboost_basic9_histstats_rolling_3fold.csv`

## Fold Windows

- fold0: train=[2011-01-01 00:00:00 ~ 2011-06-19 23:00:00] (2736), valid=[2011-07-01 00:00:00 ~ 2011-12-19 23:00:00] (2736), rmsle=0.447657
- fold1: train=[2011-01-01 00:00:00 ~ 2011-12-19 23:00:00] (5472), valid=[2012-01-01 00:00:00 ~ 2012-06-19 23:00:00] (2736), rmsle=0.435793
- fold2: train=[2011-01-01 00:00:00 ~ 2012-06-19 23:00:00] (8208), valid=[2012-07-01 00:00:00 ~ 2012-12-19 23:00:00] (2736), rmsle=0.456434
