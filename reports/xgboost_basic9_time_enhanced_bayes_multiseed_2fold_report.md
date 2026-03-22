# XGBoost Basic9 Time-Enhanced Bayesian Multi-Seed 2-Fold Report

- Time: 2026-03-22 23:22:31
- Input: `cv2_random_train_miss_train_20260322_205429.csv`
- Feature strategy: `base9 + datetime_derived_time_features + bayesian_hpo_multiseed`
- Seeds: `[42, 2024, 3407]`
- Trials per seed: `120`
- Selection rule: `sort by (mean_rmsle asc, std_rmsle asc)`

## Seed Results
- seed=3407 | mean=0.341124 | std=0.003578 | folds=0.344701, 0.337546
- seed=2024 | mean=0.341975 | std=0.002376 | folds=0.344351, 0.339599
- seed=42 | mean=0.342020 | std=0.002326 | folds=0.344345, 0.339694

- Selected seed: `3407`
- Final fold scores (RMSLE): `0.344701, 0.337546`
- Final mean RMSLE: `0.341124`
- Final std RMSLE: `0.003578`
- Models: `xgboost_basic9_time_enhanced_bayes_multiseed_2fold_seed3407_fold0.json, xgboost_basic9_time_enhanced_bayes_multiseed_2fold_seed3407_fold1.json, xgboost_basic9_time_enhanced_bayes_multiseed_2fold_full.json`
- OOF: `oof_xgboost_basic9_time_enhanced_bayes_multiseed_2fold.csv`
