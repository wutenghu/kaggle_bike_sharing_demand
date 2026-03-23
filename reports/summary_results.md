# Summary Results

## Reports (Newest First)
- 2026-03-23 17:54:38 - `randomforest_bucketstats_trainonly_2fold_report.md`
  均值 RMSLE：`0.415526`。
- 2026-03-23 17:54:31 - `xgboost_bucketstats_trainonly_2fold_report.md`
  均值 RMSLE：`0.390531`。
- 2026-03-23 16:51:00 - `basic9_model_compare_train_miss_train.md`
  RMSLE 对比：XGBoost `0.419085`，RandomForest `0.447828`（越小越好）。
- 2026-03-23 16:50:09 - `randomforest_basic_2fold_report.md`
  均值 RMSLE：`0.447828`。
- 2026-03-23 16:49:52 - `xgboost_basic_2fold_report.md`
  均值 RMSLE：`0.419085`。
- 2026-03-23 00:46:42 - `xgboost_basic10_y_s_w_h_stat_2fold_report.md`
  均值 RMSLE：`0.324108`。
- 2026-03-23 00:31:28 - `xgboost_basic9_histstats_timeprefix_2fold_calibration_report.md`
  均值 RMSLE：`0.307747 -> 0.307366`。
- 2026-03-23 00:21:49 - `xgboost_basic9_histstats_timeprefix_2fold_report.md`
  均值 RMSLE：`0.307741`。
- 2026-03-23 00:08:35 - `xgboost_basic9_histstats_rolling_3fold_report.md`
  时间滚动 3-fold 均值 RMSLE：`0.446628`。
- 2026-03-23 00:05:12 - `xgboost_basic9_time_enhanced_rolling_3fold_report.md`
  时间滚动 3-fold 均值 RMSLE：`0.524782`（对比 `0.523644` 未提升）。
- 2026-03-22 23:59:13 - `xgboost_basic9_time_rolling_3fold_report.md`
  时间滚动 3-fold 均值 RMSLE：`0.523644`。
- 2026-03-22 23:22:31 - `xgboost_basic9_time_enhanced_bayes_multiseed_2fold_report.md`
  均值 RMSLE：`0.341124`。
- 2026-03-22 22:40:45 - `xgboost_basic9_time_enhanced_bayes_2fold_report.md`
  均值 RMSLE：`0.344681 -> 0.345713`（回退）。
- 2026-03-22 22:34:05 - `xgboost_basic9_time_enhanced_2fold_report.md`
  均值 RMSLE：`0.382024 -> 0.344681`。
- 2026-03-22 22:20:27 - `xgboost_basic9_bayes_2fold_report.md`
  均值 RMSLE：`0.388607 -> 0.382024`。
- 2026-03-22 22:14:43 - `xgboost_basic9_top5cross_bayes_2fold_report.md`
  均值 RMSLE：`0.390383 -> 0.383569`。
- 2026-03-22 22:04:19 - `xgboost_basic9_top5cross_2fold_report.md`
  均值 RMSLE：`0.390383`（相较基础 9 特征退化）。
- 2026-03-22 21:06:51 - `xgboost_basic_2fold_report.md`
  均值 RMSLE：`0.388607`。

## 2-Fold CV Summary

| metrics_file | model | features | fold_scores | mean_rmsle | model_files |
|---|---|---|---|---:|---|
| `xgboost_basic10_y_s_w_h_stat_2fold_metrics.json` | `xgboost.XGBRegressor` | `year + season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour + mean(count&#124;year,season,workingday,hour)` | `0.322110, 0.326107` | 0.324108 | `xgboost_basic10_y_s_w_h_stat_2fold_fold0.json, xgboost_basic10_y_s_w_h_stat_2fold_fold1.json, xgboost_basic10_y_s_w_h_stat_2fold_full.json` |
| `xgboost_basic9_histstats_timeprefix_2fold_calibration_metrics.json` | `xgboost post-calibration (OOF)` | `base9 + leakage_safe_historical_priors_timeprefix + year_month_logshift_calibration` | `N/A` | 0.307366 | `xgboost_basic9_histstats_timeprefix_2fold_full.json` |
| `xgboost_basic9_histstats_timeprefix_2fold_metrics.json` | `xgboost.XGBRegressor` | `base9 + leakage_safe_historical_priors_timeprefix` | `0.309743, 0.305739` | 0.307741 | `xgboost_basic9_histstats_timeprefix_2fold_fold0.json, xgboost_basic9_histstats_timeprefix_2fold_fold1.json, xgboost_basic9_histstats_timeprefix_2fold_full.json` |
| `xgboost_basic9_time_enhanced_bayes_multiseed_2fold_metrics.json` | `xgboost.XGBRegressor (Optuna TPE, seeds=[42,2024,3407], 120 trials/seed)` | `base9 + datetime_derived_time_features + bayesian_hpo_multiseed` | `0.344701, 0.337546` | 0.341124 | `xgboost_basic9_time_enhanced_bayes_multiseed_2fold_seed3407_fold0.json, xgboost_basic9_time_enhanced_bayes_multiseed_2fold_seed3407_fold1.json, xgboost_basic9_time_enhanced_bayes_multiseed_2fold_full.json` |
| `xgboost_basic9_time_enhanced_bayes_2fold_metrics.json` | `xgboost.XGBRegressor (Optuna TPE, 40 trials)` | `base9 + datetime_derived_time_features + bayesian_hpo` | `0.346857, 0.344568` | 0.345713 | `xgboost_basic9_time_enhanced_bayes_2fold_fold0.json, xgboost_basic9_time_enhanced_bayes_2fold_fold1.json, xgboost_basic9_time_enhanced_bayes_2fold_full.json` |
| `xgboost_basic9_time_enhanced_2fold_metrics.json` | `xgboost.XGBRegressor` | `base9 + datetime_derived_time_features` | `0.348446, 0.340917` | 0.344681 | `xgboost_basic9_time_enhanced_2fold_full.json` |
| `xgboost_basic9_bayes_2fold_metrics.json` | `xgboost.XGBRegressor (Optuna TPE, 40 trials)` | `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour + bayesian_hpo` | `0.384739, 0.379308` | 0.382024 | `xgboost_basic9_bayes_2fold_fold0.json, xgboost_basic9_bayes_2fold_fold1.json, xgboost_basic9_bayes_2fold_full.json` |
| `xgboost_basic9_top5cross_bayes_2fold_metrics.json` | `xgboost.XGBRegressor (Optuna TPE, 40 trials)` | `base9 + top5 gain cross(10) + bayesian_hpo` | `0.384412, 0.382726` | 0.383569 | `xgboost_basic9_top5cross_bayes_2fold_fold0.json, xgboost_basic9_top5cross_bayes_2fold_fold1.json, xgboost_basic9_top5cross_bayes_2fold_full.json` |
| `xgboost_basic9_top5cross_2fold_metrics.json` | `xgboost.XGBRegressor` | `base9 + top5 gain cross(10 products)` | `0.392843, 0.387924` | 0.390383 | `xgboost_basic9_top5cross_2fold_full.json` |
| `xgboost_basic_2fold_metrics.json` | `xgboost.XGBRegressor` | `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour` | `0.417473, 0.420696` | 0.419085 | `xgboost_basic_9feat_hour_fold0.json, xgboost_basic_9feat_hour_fold1.json, xgboost_basic_9feat_hour_full.json, xgboost_basic_9feat_hour_full.pkl` |
| `randomforest_basic_2fold_metrics.json` | `sklearn.ensemble.RandomForestRegressor` | `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour` | `0.451583, 0.444073` | 0.447828 | `randomforest_basic_9feat_hour_fold0.pkl, randomforest_basic_9feat_hour_fold1.pkl, randomforest_basic_9feat_hour_full.pkl` |
| `xgboost_bucketstats_trainonly_2fold_metrics.json` | `xgboost.XGBRegressor` | `season,month,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour + stat_count_mean/std/q05/q25/q50/q75/q95` | `0.391936, 0.389127` | 0.390531 | `xgboost_bucketstats_trainonly_2fold_fold0.json, xgboost_bucketstats_trainonly_2fold_fold1.json, xgboost_bucketstats_trainonly_2fold_full.json, xgboost_bucketstats_trainonly_2fold_full.pkl` |
| `randomforest_bucketstats_trainonly_2fold_metrics.json` | `sklearn.ensemble.RandomForestRegressor` | `season,month,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour + stat_count_mean/std/q05/q25/q50/q75/q95` | `0.415225, 0.415828` | 0.415526 | `randomforest_bucketstats_trainonly_2fold_fold0.pkl, randomforest_bucketstats_trainonly_2fold_fold1.pkl, randomforest_bucketstats_trainonly_2fold_full.pkl` |

## Time-Rolling 3-Fold CV Summary

| metrics_file | model | features | fold_scores | mean_rmsle | model_files |
|---|---|---|---|---:|---|
| `xgboost_basic9_histstats_rolling_3fold_metrics.json` | `xgboost.XGBRegressor` | `base9 + leakage_safe_historical_mean_features` | `0.447657, 0.435793, 0.456434` | 0.446628 | `xgboost_basic9_histstats_rolling_3fold_fold0.json, xgboost_basic9_histstats_rolling_3fold_fold1.json, xgboost_basic9_histstats_rolling_3fold_fold2.json, xgboost_basic9_histstats_rolling_3fold_full.json` |
| `xgboost_basic9_time_enhanced_rolling_3fold_metrics.json` | `xgboost.XGBRegressor` | `base9 + datetime_derived_time_features + hour_cross_features` | `0.422233, 0.670099, 0.482012` | 0.524782 | `xgboost_basic9_time_enhanced_rolling_3fold_fold0.json, xgboost_basic9_time_enhanced_rolling_3fold_fold1.json, xgboost_basic9_time_enhanced_rolling_3fold_fold2.json, xgboost_basic9_time_enhanced_rolling_3fold_full.json` |
| `xgboost_basic9_time_rolling_3fold_metrics.json` | `xgboost.XGBRegressor` | `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour` | `0.428235, 0.653881, 0.488816` | 0.523644 | `xgboost_basic9_time_rolling_3fold_fold0.json, xgboost_basic9_time_rolling_3fold_fold1.json, xgboost_basic9_time_rolling_3fold_fold2.json, xgboost_basic9_time_rolling_3fold_full.json` |
