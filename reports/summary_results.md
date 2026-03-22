# Summary Results

## Reports (Newest First)
- 2026-03-23 00:46:42 - `xgboost_basic10_y_s_w_h_stat_2fold_report.md`
  核心改动：在随机 `2-fold` 口径下，将基础特征扩展为 `year + base9`，并新增 `year*season*workingday*hour` 组合的 `count` 统计特征（仅用训练折中 `source=train` 样本聚合）。该设置下均值 RMSLE 为 0.324108。
- 2026-03-23 00:31:28 - `xgboost_basic9_histstats_timeprefix_2fold_calibration_report.md`
  核心改动：在 `2-fold` OOF 上按 `month` 与 `year-month` 学习轻量 log-shift 校准系数（等价于乘性校准），评估系统性时段偏差。`year-month` 校准将均值 RMSLE 从 0.307747 降至 0.307366，并将 2011/2012 年度 log 偏差收敛到接近 0。
- 2026-03-23 00:21:49 - `xgboost_basic9_histstats_timeprefix_2fold_report.md`
  核心改动：保持随机 `2-fold` 切分不变，在 `base9` 上加入严格按时间前缀计算的历史统计先验（训练样本 past-only、验证样本仅使用训练折中早于该时刻的统计）。在不使用 test 不可得 `count` 特征的前提下，均值 RMSLE 降至 0.307741。
- 2026-03-23 00:08:35 - `xgboost_basic9_histstats_rolling_3fold_report.md`
  核心改动：在 `base9` 上增加严格防泄漏的历史统计特征（`hist_mean_global/hour/dayofweek/month/hour_x_workingday/hour_x_season`），训练段仅使用过去样本，验证段仅映射训练段统计。时间滚动 `3-fold` 均值 RMSLE 降至 0.446628，且折间波动显著收敛。
- 2026-03-23 00:05:12 - `xgboost_basic9_time_enhanced_rolling_3fold_report.md`
  核心改动：在 `base9` 上加入时间增强特征（`dayofweek/month/dayofyear`、月初月末、`sin/cos` 周期编码、`hour×workingday`、`hour×season`），并沿用相同时间滚动 `3-fold` 口径评估。结果均值 RMSLE 为 0.524782，较纯 `base9`（0.523644）未提升。
- 2026-03-22 23:59:13 - `xgboost_basic9_time_rolling_3fold_report.md`
  核心改动：基于 `cv2_random_train_miss_train_20260322_205429.csv` 改为时间滚动扩展窗口 `3-fold`（4段连续时间块，后3段依次做验证）评估 `base9`。结果均值 RMSLE 升至 0.523644，验证了随机 `2-fold` 对时序泛化存在显著乐观偏差。
- 2026-03-22 23:22:31 - `xgboost_basic9_time_enhanced_bayes_multiseed_2fold_report.md`
  核心改动：固定 `base9 + datetime_derived_time_features`，执行多 seed（`42/2024/3407`）且每个 seed `120` trials 的贝叶斯优化，并按 `(mean_rmsle, std_rmsle)` 选最稳最优。最终选中 `seed=3407`，均值 RMSLE 下降到 0.341124。
- 2026-03-22 22:40:45 - `xgboost_basic9_time_enhanced_bayes_2fold_report.md`
  核心改动：在 `base9 + datetime_derived_time_features` 口径上进一步使用贝叶斯优化（Optuna TPE，40 trials）搜索超参数，并按固定随机 `2-fold` 复训。均值 RMSLE 从 0.344681 提升到 0.345713（小幅回退，说明默认参数已较优）。
- 2026-03-22 22:34:05 - `xgboost_basic9_time_enhanced_2fold_report.md`
  核心改动：仅在 9 个基础特征上新增可由 `datetime` 直接构造的时间增强特征（`dayofweek/month`、周期编码、周末/高峰标记），不改切分协议。`2-fold` 均值 RMSLE 从 0.382024 下降到 0.344681。
- 2026-03-22 22:20:27 - `xgboost_basic9_bayes_2fold_report.md`
  核心改动：基于 `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour` 九个基础特征执行贝叶斯优化（Optuna TPE，40 trials），并在固定随机 `2-fold` 上复训验证。均值 RMSLE 从 0.388607 提升到 0.382024。
- 2026-03-22 22:14:43 - `xgboost_basic9_top5cross_bayes_2fold_report.md`
  核心改动：在 `base9 + top5交叉(10)` 特征上使用贝叶斯优化（Optuna TPE，40 trials）搜索 XGBoost 超参数，并按固定随机 `2-fold` 复训。均值 RMSLE 从 0.390383 提升到 0.383569。
- 2026-03-22 22:04:19 - `xgboost_basic9_top5cross_2fold_report.md`
  核心改动：在 9 个基础特征中按 gain 选取 top5（`hour, workingday, season, temp, atemp`），两两乘积生成 10 个交叉高维特征并做 `2-fold` 对照。结果相较基础 9 特征略有退化，暂不作为默认方案。
- 2026-03-22 21:06:51 - `xgboost_basic_2fold_report.md`
  核心改动：在基础特征上新增 `hour`，使用 `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour` 训练 XGBoost 并按随机 `2-fold`（`cv_fold`）评估 `count`。同步更新 fold 模型、全量模型和 OOF 结果。

## 2-Fold CV Summary

| metrics_file | model | features | fold_scores | mean_rmsle | model_files |
|---|---|---|---|---:|---|
| `xgboost_basic10_y_s_w_h_stat_2fold_metrics.json` | `xgboost.XGBRegressor` | `year + season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour + mean(count|year,season,workingday,hour)` | `0.322110, 0.326107` | 0.324108 | `xgboost_basic10_y_s_w_h_stat_2fold_fold0.json, xgboost_basic10_y_s_w_h_stat_2fold_fold1.json, xgboost_basic10_y_s_w_h_stat_2fold_full.json` |
| `xgboost_basic9_histstats_timeprefix_2fold_calibration_metrics.json` | `xgboost post-calibration (OOF)` | `base9 + leakage_safe_historical_priors_timeprefix + year_month_logshift_calibration` | `N/A` | 0.307366 | `xgboost_basic9_histstats_timeprefix_2fold_full.json` |
| `xgboost_basic9_histstats_timeprefix_2fold_metrics.json` | `xgboost.XGBRegressor` | `base9 + leakage_safe_historical_priors_timeprefix` | `0.309743, 0.305739` | 0.307741 | `xgboost_basic9_histstats_timeprefix_2fold_fold0.json, xgboost_basic9_histstats_timeprefix_2fold_fold1.json, xgboost_basic9_histstats_timeprefix_2fold_full.json` |
| `xgboost_basic9_time_enhanced_bayes_multiseed_2fold_metrics.json` | `xgboost.XGBRegressor (Optuna TPE, seeds=[42,2024,3407], 120 trials/seed)` | `base9 + datetime_derived_time_features + bayesian_hpo_multiseed` | `0.344701, 0.337546` | 0.341124 | `xgboost_basic9_time_enhanced_bayes_multiseed_2fold_seed3407_fold0.json, xgboost_basic9_time_enhanced_bayes_multiseed_2fold_seed3407_fold1.json, xgboost_basic9_time_enhanced_bayes_multiseed_2fold_full.json` |
| `xgboost_basic9_time_enhanced_bayes_2fold_metrics.json` | `xgboost.XGBRegressor (Optuna TPE, 40 trials)` | `base9 + datetime_derived_time_features + bayesian_hpo` | `0.346857, 0.344568` | 0.345713 | `xgboost_basic9_time_enhanced_bayes_2fold_fold0.json, xgboost_basic9_time_enhanced_bayes_2fold_fold1.json, xgboost_basic9_time_enhanced_bayes_2fold_full.json` |
| `xgboost_basic9_time_enhanced_2fold_metrics.json` | `xgboost.XGBRegressor` | `base9 + datetime_derived_time_features` | `0.348446, 0.340917` | 0.344681 | `xgboost_basic9_time_enhanced_2fold_full.json` |
| `xgboost_basic9_bayes_2fold_metrics.json` | `xgboost.XGBRegressor (Optuna TPE, 40 trials)` | `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour + bayesian_hpo` | `0.384739, 0.379308` | 0.382024 | `xgboost_basic9_bayes_2fold_fold0.json, xgboost_basic9_bayes_2fold_fold1.json, xgboost_basic9_bayes_2fold_full.json` |
| `xgboost_basic9_top5cross_bayes_2fold_metrics.json` | `xgboost.XGBRegressor (Optuna TPE, 40 trials)` | `base9 + top5 gain cross(10) + bayesian_hpo` | `0.384412, 0.382726` | 0.383569 | `xgboost_basic9_top5cross_bayes_2fold_fold0.json, xgboost_basic9_top5cross_bayes_2fold_fold1.json, xgboost_basic9_top5cross_bayes_2fold_full.json` |
| `xgboost_basic9_top5cross_2fold_metrics.json` | `xgboost.XGBRegressor` | `base9 + top5 gain cross(10 products)` | `0.392843, 0.387924` | 0.390383 | `xgboost_basic9_top5cross_2fold_full.json` |
| `xgboost_basic_2fold_metrics.json` | `xgboost.XGBRegressor` | `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour` | `0.393041, 0.384173` | 0.388607 | `xgboost_basic_9feat_hour_fold0.json, xgboost_basic_9feat_hour_fold1.json, xgboost_basic_9feat_hour_full.json` |

## Time-Rolling 3-Fold CV Summary

| metrics_file | model | features | fold_scores | mean_rmsle | model_files |
|---|---|---|---|---:|---|
| `xgboost_basic9_histstats_rolling_3fold_metrics.json` | `xgboost.XGBRegressor` | `base9 + leakage_safe_historical_mean_features` | `0.447657, 0.435793, 0.456434` | 0.446628 | `xgboost_basic9_histstats_rolling_3fold_fold0.json, xgboost_basic9_histstats_rolling_3fold_fold1.json, xgboost_basic9_histstats_rolling_3fold_fold2.json, xgboost_basic9_histstats_rolling_3fold_full.json` |
| `xgboost_basic9_time_enhanced_rolling_3fold_metrics.json` | `xgboost.XGBRegressor` | `base9 + datetime_derived_time_features + hour_cross_features` | `0.422233, 0.670099, 0.482012` | 0.524782 | `xgboost_basic9_time_enhanced_rolling_3fold_fold0.json, xgboost_basic9_time_enhanced_rolling_3fold_fold1.json, xgboost_basic9_time_enhanced_rolling_3fold_fold2.json, xgboost_basic9_time_enhanced_rolling_3fold_full.json` |
| `xgboost_basic9_time_rolling_3fold_metrics.json` | `xgboost.XGBRegressor` | `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour` | `0.428235, 0.653881, 0.488816` | 0.523644 | `xgboost_basic9_time_rolling_3fold_fold0.json, xgboost_basic9_time_rolling_3fold_fold1.json, xgboost_basic9_time_rolling_3fold_fold2.json, xgboost_basic9_time_rolling_3fold_full.json` |
