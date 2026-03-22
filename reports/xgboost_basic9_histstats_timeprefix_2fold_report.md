# XGBoost Basic9 HistStats TimePrefix 2-Fold Report

- Time: 2026-03-23 00:21:49
- Input: `cv2_random_train_miss_train_20260322_205429.csv`
- Feature strategy: `base9 + leakage_safe_historical_priors_timeprefix`
- Added features: `hist_mean_global, hist_mean_hour, hist_mean_dayofweek, hist_mean_month, hist_mean_hour_x_workingday, hist_mean_hour_x_season`
- CV strategy: `random_2fold_from_cv_fold`
- Hist rule: `train uses past-only stats; valid uses train-only prefix stats (datetime strict)`
- Fold RMSLE: `0.309743, 0.305739`
- Mean RMSLE: `0.307741`
- Std RMSLE: `0.002002`
- Models: `xgboost_basic9_histstats_timeprefix_2fold_fold0.json, xgboost_basic9_histstats_timeprefix_2fold_fold1.json, xgboost_basic9_histstats_timeprefix_2fold_full.json`
- OOF: `oof_xgboost_basic9_histstats_timeprefix_2fold.csv`
