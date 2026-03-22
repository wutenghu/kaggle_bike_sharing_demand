# XGBoost TSFresh 2-Fold CV Report

- Time: 2026-03-22 21:19:07
- Input: `cv2_random_train_miss_train_20260322_205429.csv`
- Base features: `season, holiday, workingday, weather, temp, atemp, humidity, windspeed, hour`
- tsfresh candidates: `777`
- tsfresh selected per fold: `120, 120`
- Fold scores (RMSLE): `0.312757, 0.307082`
- Mean RMSLE: `0.309919`
- Models: `xgboost_tsfresh_2fold_fold0.json, xgboost_tsfresh_2fold_fold1.json, xgboost_tsfresh_2fold_full.json`
- OOF: `oof_xgboost_tsfresh_2fold.csv`
