# XGBoost Basic10 + YearSeasonWorkingdayHour Stat 2-Fold Report

- Time: 2026-03-23 00:46:42
- Input: `cv2_random_train_miss_train_20260322_205429.csv`
- Feature strategy: `base10(year+base9) + mean(count|year,season,workingday,hour) from source=train`
- Added feature: `stat_mean_count_year_season_workingday_hour`
- Group stat rule: `statistics computed using only source=train rows in each training fold`
- CV strategy: `random_2fold_from_cv_fold`
- Fold RMSLE: `0.322110, 0.326107`
- Mean RMSLE: `0.324108`
- Std RMSLE: `0.001999`
- Models: `xgboost_basic10_y_s_w_h_stat_2fold_fold0.json, xgboost_basic10_y_s_w_h_stat_2fold_fold1.json, xgboost_basic10_y_s_w_h_stat_2fold_full.json`
- OOF: `oof_xgboost_basic10_y_s_w_h_stat_2fold.csv`
