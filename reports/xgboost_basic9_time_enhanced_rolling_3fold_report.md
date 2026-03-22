# XGBoost Basic9 Time-Enhanced Rolling 3-Fold Report

- Time: 2026-03-23 00:05:12
- Input: `cv2_random_train_miss_train_20260322_205429.csv`
- Feature strategy: `base9 + datetime_derived_time_features + hour_cross_features`
- Added features: `dayofweek, month, dayofyear, is_month_start, is_month_end, hour_sin, hour_cos, month_sin, month_cos, dayofyear_sin, dayofyear_cos, hour_x_workingday, hour_x_season`
- CV strategy: `time_rolling_expanding_window_3fold`
- Fold definition: `split sorted samples into 4 contiguous blocks; folds validate on block 2/3/4`
- Fold RMSLE: `0.422233, 0.670099, 0.482012`
- Mean RMSLE: `0.524782`
- Std RMSLE: `0.105614`
- Params source: `xgboost_basic9_bayes_2fold_metrics.json`
- Models: `xgboost_basic9_time_enhanced_rolling_3fold_fold0.json, xgboost_basic9_time_enhanced_rolling_3fold_fold1.json, xgboost_basic9_time_enhanced_rolling_3fold_fold2.json, xgboost_basic9_time_enhanced_rolling_3fold_full.json`
- OOF: `oof_xgboost_basic9_time_enhanced_rolling_3fold.csv`

## Fold Windows

- fold0: train=[2011-01-01 00:00:00 ~ 2011-06-19 23:00:00] (2736), valid=[2011-07-01 00:00:00 ~ 2011-12-19 23:00:00] (2736), rmsle=0.422233
- fold1: train=[2011-01-01 00:00:00 ~ 2011-12-19 23:00:00] (5472), valid=[2012-01-01 00:00:00 ~ 2012-06-19 23:00:00] (2736), rmsle=0.670099
- fold2: train=[2011-01-01 00:00:00 ~ 2012-06-19 23:00:00] (8208), valid=[2012-07-01 00:00:00 ~ 2012-12-19 23:00:00] (2736), rmsle=0.482012
