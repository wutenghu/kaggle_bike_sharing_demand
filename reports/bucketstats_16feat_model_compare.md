# BucketStats 16-Feature Model Compare

## Data & Features
- Train filter: `source in (train, miss_train)`
- Predict target: `source=test` aligned by `sampleSubmission_expected.csv`
- Features: `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour,stat_count_mean_y_m_w_h,stat_count_std_y_m_w_h,stat_count_q05_y_m_w_h,stat_count_q25_y_m_w_h,stat_count_q50_y_m_w_h,stat_count_q75_y_m_w_h,stat_count_q95_y_m_w_h`

## 2-Fold CV (Fixed Split)
- Split file: `data/train_cv2fold_split.csv`
- XGBoost RMSLE folds: `0.358667, 0.361891`; mean `0.360279`
- RandomForest RMSLE folds: `0.382805, 0.373539`; mean `0.378172`

## Model Files
- `models/xgboost_bucketstats_16feat_full.pkl`
- `models/xgboost_bucketstats_16feat_full.json`
- `models/randomforest_bucketstats_16feat_full.pkl`

## Submission Files
- `outputs/submission_xgboost_bucketstats_16feat_full.csv`
- `outputs/submission_randomforest_bucketstats_16feat_full.csv`

## Alignment Checks
- XGBoost: rows `6493`, datetime exact match `True`, integer/non-negative/no-NA `True/True/True`
- RandomForest: rows `6493`, datetime exact match `True`, integer/non-negative/no-NA `True/True/True`
