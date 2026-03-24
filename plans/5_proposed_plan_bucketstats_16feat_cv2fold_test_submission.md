<proposed_plan>
### 基于 Bucket-Stats + 固定 2-Fold（`datetime` 对齐）的双模型对比实验计划

#### Summary
基于 `data/train_test_merged_with_bucket_stats.csv` 与 `data/train_cv2fold_split.csv`，开展下一轮 XGBoost + RandomForest 双模型对比实验。  
本轮固定使用 `datetime` 对齐的 2-fold 切分，保证与后续实验严格可比；产物采用新前缀，避免覆盖现有 baseline 文件。

#### Implementation Changes
1. 输入与切分
- 训练数据：`data/train_test_merged_with_bucket_stats.csv`
- 切分数据：`data/train_cv2fold_split.csv`
- 对齐键：`datetime`（仅按 `datetime` 映射 `cv_fold`）
- 训练样本：`source in (train, miss_train)`，标签列 `count`

2. 特征与模型
- 特征：`base9 + 7个桶统计特征`（共 16 维）  
  - base9：`season,holiday,workingday,weather,temp,atemp,humidity,windspeed,hour`
  - stats7：`stat_count_mean_y_m_w_h,stat_count_std_y_m_w_h,stat_count_q05_y_m_w_h,stat_count_q25_y_m_w_h,stat_count_q50_y_m_w_h,stat_count_q75_y_m_w_h,stat_count_q95_y_m_w_h`
- 模型：`XGBoost`、`RandomForest`
- 评估：固定 2-fold CV（复用 `data/train_cv2fold_split.csv`）

3. 产物命名（不覆盖旧文件）
- 指标：`outputs/bucketstats_16feat_metrics.json`
- 模型：
  - `models/xgboost_bucketstats_16feat_full.pkl`（并保留 `json` 兼容产物）
  - `models/randomforest_bucketstats_16feat_full.pkl`
- 提交文件：
  - `outputs/submission_xgboost_bucketstats_16feat_full.csv`
  - `outputs/submission_randomforest_bucketstats_16feat_full.csv`
- 报告：`reports/bucketstats_16feat_model_compare.md`

4. 报告追加
- 在 `reports/summary_results.md` 的 `## Reports (Newest First)` 追加本轮摘要
- 在 `reports/summary_results.md` 的 `## 2-Fold CV Summary` 追加本轮双模型条目
- 追加时保持现有文档风格与格式，不改已有章节结构

#### Public Interfaces / Data Contract
- 读取：`data/train_test_merged_with_bucket_stats.csv`、`data/train_cv2fold_split.csv`
- 写出：`outputs/*bucketstats_16feat*`、`models/*bucketstats_16feat*`、`reports/bucketstats_16feat_model_compare.md`
- `cv_fold` 对齐规则：仅按 `datetime` 一对一映射，取值限定为 `0/1`

#### Test Plan
1. 切分一致性
- `train/miss_train` 全量样本都能按 `datetime` 命中 `cv_fold`
- `cv_fold` 仅包含 `0/1`，无空值

2. 训练评估
- 两模型均输出 fold RMSLE 与 mean RMSLE
- metrics 中记录正确的特征列表与 split 路径

3. 提交约束
- 两份 submission 列名严格为 `datetime,count`
- 行数均为 `6493`，`datetime` 与模板完全对齐
- `count` 全为非负整数且无空值

4. 变更边界
- `basic_9feat` 历史产物不被覆盖
- 仅新增本轮 bucketstats 前缀产物与报告内容

#### Assumptions
- `data/train_cv2fold_split.csv` 中 `datetime` 可唯一映射训练样本
- 本轮仅做 bucket-stats 特征集下的双模型对比，不包含超参搜索
</proposed_plan>
