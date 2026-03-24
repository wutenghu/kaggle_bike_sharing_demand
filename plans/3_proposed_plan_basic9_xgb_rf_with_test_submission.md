<proposed_plan>
### 基础双模型 Baseline + `source=test` 预测产出（对齐提交模板）

#### Summary
在上一版计划基础上，产出仅保留两个基础模型（XGBoost、RandomForest）的 `source=test` 预测提交文件，不生成模型对比明细 CSV。  
预测结果为整数型 `count`，并严格对齐 [sampleSubmission_expected.csv](/Users/wutenghu/Pycharm/kaggle_bike_sharing_demand/data/sampleSubmission_expected.csv) 的时间戳顺序与 6493 行数。

#### Implementation Changes
1. 保留 baseline 训练流程：
- 训练集：`source in (train, miss_train)`  
- 特征：`season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour`
- 指标：RMSLE（2-fold 随机 CV）
- 模型：XGBoost + RandomForest

2. 固化并落盘 2-fold 切分（保证后续严格可比较）：
- 对 `data/train_test_merged.csv` 中 `source in (train, miss_train)` 的样本按固定随机种子生成 `cv_fold`（0/1）。
- 输出切分清单：`data/train_cv2fold_split.csv`（包含 `datetime,cv_fold`，可附带主键列）。
- 后续所有 2-fold CV 训练必须复用该切分文件，不再临时重新随机划分。

3. 统一 full 模型持久化为 `pkl`：
- XGBoost：新增 `models/xgboost_basic_9feat_full.pkl`（保留现有 `.json` 兼容产物）。
- RandomForest：`models/randomforest_basic_9feat_full.pkl`。

4. 新增 test 预测脚本（`src/predict_test_basic_models.py`）：
- 输入：`train_test_merged.csv`（`source=test`）、`sampleSubmission_expected.csv`、两个 full 模型 `pkl`。
- 预测流程：按同一特征顺序推理，`clip >= 0`，`round` 后转整数。
- 对齐流程：以 `sampleSubmission_expected.csv` 为主表按 `datetime` 对齐，确保顺序与行数完全一致。
 - 若因上游补齐导致 `source=test` 有极少缺失键，允许回退填充值（中位数）兜底，最终输出不得含空值。

5. 预测文件交付（仅这两份）：
- `outputs/submission_xgboost_basic_9feat_full.csv`（`datetime,count`）
- `outputs/submission_randomforest_basic_9feat_full.csv`（`datetime,count`）

6. 报告补充：
- 在 `reports/basic_9feat_model_compare.md` 记录两份 submission 文件路径和对齐校验结果。
- 在 `reports/summary_results.md` 中记录实验的结论（注意在原有基础上增加内容，不要改变报告的格式，不要增加章节）

7. 运行环境约束（新增）：
- 为避免 OpenMP 共享内存限制导致训练中断，默认单线程执行：
  - 模型参数：`n_jobs=1`
  - 或运行时变量：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`

#### Public Interfaces / Artifacts
- 新增：`src/predict_test_basic_models.py`
- 新增：`data/train_cv2fold_split.csv`
- 新增：`models/xgboost_basic_9feat_full.pkl`
- 新增：`models/randomforest_basic_9feat_full.pkl`
- 新增：两个 submission 预测文件（`outputs/`）

#### Test Plan
1. 模型文件可用性：
- 两个 `full.pkl` 均可加载并预测。

2. 预测文件结构与对齐：
- 两个 submission 文件列名严格为 `datetime,count`。
- 行数均 `6493`，`datetime` 序列与 `sampleSubmission_expected.csv` 完全一致。

3. 预测值约束：
- `count` 全为整数、非负、无空值。

4. 训练评估：
- XGBoost 与 RandomForest 均输出 2-fold RMSLE 与 mean RMSLE 到各自 metrics/report。
- 总览报告包含两模型评估结果与 submission 文件路径。
5. 切分复用一致性：
- 生成并保存 `data/train_cv2fold_split.csv`，`cv_fold` 仅包含 0/1。
- 当次训练与后续复训均基于该文件切分；重复运行时每条样本 `cv_fold` 不变。

6. 当前基线结果（用于回归对比）：
- XGBoost mean RMSLE
- RandomForest mean RMSLE

#### Assumptions
- `source=test` 在 `train_test_merged.csv` 可与模板时间戳完全对齐；若存在缺失，以模板为准并要求最终输出无缺失预测。
- 当前阶段目标是稳定 baseline 与可提交产物，不做超参搜索。
</proposed_plan>
