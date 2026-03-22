# XGBoost Basic9 + Top5 Cross 2-Fold Report

- Time: 2026-03-22 22:04:19
- Input: `cv2_random_train_miss_train_20260322_205429.csv`
- Top5 features by gain: `hour, workingday, season, temp, atemp`
- Added cross features: `10`
- Baseline fold RMSLE: `0.393041, 0.384173`
- Baseline mean RMSLE: `0.388607`
- Cross fold RMSLE: `0.392843, 0.387924`
- Cross mean RMSLE: `0.390383`
- Delta (cross - baseline): `0.001776`
- Full model: `xgboost_basic9_top5cross_2fold_full.json`
- OOF: `oof_xgboost_basic9_top5cross_2fold.csv`
