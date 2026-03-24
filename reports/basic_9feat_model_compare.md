# Basic 9-Feature Model Compare

## Data & Features
- Train filter: `source in (train, miss_train)`
- Predict target: `source=test` aligned by `sampleSubmission_expected.csv`
- Features: `season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour`

## 2-Fold CV (Fixed Split)
- Split file: `data/train_cv2fold_split.csv`
- XGBoost RMSLE folds: `0.574306, 0.570711`; mean `0.572509`
- RandomForest RMSLE folds: `0.476211, 0.461948`; mean `0.469080`

## Model Files
- `models/xgboost_basic_9feat_full.pkl`
- `models/xgboost_basic_9feat_full.json`
- `models/randomforest_basic_9feat_full.pkl`

## Submission Files
- `outputs/submission_xgboost_basic_9feat_full.csv`
- `outputs/submission_randomforest_basic_9feat_full.csv`

## Alignment Checks
- XGBoost: rows `6493`, datetime exact match `True`, integer/non-negative/no-NA `True/True/True`
- RandomForest: rows `6493`, datetime exact match `True`, integer/non-negative/no-NA `True/True/True`
