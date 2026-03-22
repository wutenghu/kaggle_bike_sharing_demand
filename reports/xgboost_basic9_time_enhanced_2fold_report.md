# XGBoost Basic9 Time-Enhanced 2-Fold Report

- Time: 2026-03-22 22:34:05
- Input: `cv2_random_train_miss_train_20260322_205429.csv`
- Added time features: `dayofweek, month, is_weekend, is_peak_hour, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos`
- Baseline fold RMSLE: `0.384739, 0.379308`
- Baseline mean RMSLE: `0.382024`
- Enhanced fold RMSLE: `0.348446, 0.340917`
- Enhanced mean RMSLE: `0.344681`
- Delta (enhanced - baseline): `-0.037342`
- Full model: `xgboost_basic9_time_enhanced_2fold_full.json`
- OOF: `oof_xgboost_basic9_time_enhanced_2fold.csv`
