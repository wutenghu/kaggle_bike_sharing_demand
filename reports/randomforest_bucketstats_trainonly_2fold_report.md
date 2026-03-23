# RandomForest BucketStats TrainOnly 2-Fold CV Report

- Time: 2026-03-23 17:54:38
- Input: `cv2_random_train_miss_train_with_bucket_stats_20260323_174324.csv`
- Training filter: `source=train`
- Features: `season, month, holiday, workingday, weather, temp, atemp, humidity, windspeed, hour, stat_count_mean_s_m_w_h, stat_count_std_s_m_w_h, stat_count_q05_s_m_w_h, stat_count_q25_s_m_w_h, stat_count_q50_s_m_w_h, stat_count_q75_s_m_w_h, stat_count_q95_s_m_w_h`
- Fold scores (RMSLE): `0.415225, 0.415828`
- Mean RMSLE: `0.415526`
- Models: `randomforest_bucketstats_trainonly_2fold_fold0.pkl, randomforest_bucketstats_trainonly_2fold_fold1.pkl, randomforest_bucketstats_trainonly_2fold_full.pkl`
- OOF: `oof_randomforest_bucketstats_trainonly_2fold.csv`
