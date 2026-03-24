# Summary Results

## Reports (Newest First)
- 2026-03-25 01:32:40 - `bucketstats_16feat_model_compare.md`
  固定2-fold（datetime 对齐）双模型：XGBoost `0.360279`，RandomForest `0.378172`；bucketstats 16 特征 submission 对齐通过。
- 2026-03-25 01:19:15 - `basic_9feat_model_compare.md`
  固定2-fold（随机种子42）双模型：XGBoost `0.572509`，RandomForest `0.469080`；basic 9 特征 submission 对齐通过。
- 2026-03-25 00:56:09 - `bucketstats_16feat_model_compare.md`
  固定2-fold（datetime 对齐）双模型：XGBoost `0.338219`，RandomForest `0.357636`；bucketstats 16 特征 submission 对齐通过。
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
| `basic_9feat_metrics.json` | `xgboost.XGBRegressor` | `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour` | `0.574306, 0.570711` | 0.572509 | `xgboost_basic_9feat_full.json, xgboost_basic_9feat_full.pkl` |
| `basic_9feat_metrics.json` | `sklearn.ensemble.RandomForestRegressor` | `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour` | `0.476211, 0.461948` | 0.469080 | `randomforest_basic_9feat_full.pkl` |
| `bucketstats_16feat_metrics.json` | `xgboost.XGBRegressor` | `base9 + stat_count_mean/std/q05/q25/q50/q75/q95 (y,m,workingday,hour)` | `0.358667, 0.361891` | 0.360279 | `xgboost_bucketstats_16feat_full.json, xgboost_bucketstats_16feat_full.pkl` |
| `bucketstats_16feat_metrics.json` | `sklearn.ensemble.RandomForestRegressor` | `base9 + stat_count_mean/std/q05/q25/q50/q75/q95 (y,m,workingday,hour)` | `0.382805, 0.373539` | 0.378172 | `randomforest_bucketstats_16feat_full.pkl` |
| `bucketstats_16feat_metrics.json` | `xgboost.XGBRegressor` | `base9 + stat_count_mean/std/q05/q25/q50/q75/q95 (y,m,workingday,hour)` | `0.335834, 0.340604` | 0.338219 | `xgboost_bucketstats_16feat_full.json, xgboost_bucketstats_16feat_full.pkl` |
