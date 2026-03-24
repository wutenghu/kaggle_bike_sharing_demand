<proposed_plan>
### 用 `year*month*workingday*hour` 桶统计特征回填四类 `source` 样本

#### Summary
将基于 `source=train` 计算得到的桶统计特征（`count_mean/count_std/count_q05/count_q25/count_q50/count_q75/count_q95`）回填到 `source in (train,test,miss_train,miss_test)` 的所有样本。  
采用“新增特征列”方式，不改动原有字段；结果保存为新文件(保留原始 `train_test_merged.csv`)。

#### Implementation Changes
1. 统计表准备（若不存在则先生成）
- 输入：`data/train_test_merged.csv`
- 过滤：`source=train`
- 分组键：`year,month,workingday,hour`（`year`，`month` 由 `datetime` 衍生）
- 统计列：`count_mean,count_std,count_q05,count_q25,count_q50,count_q75,count_q95`
- 输出：`data/count_quantiles_train_year_month_workingday_hour.csv`

2. 回填脚本（新增）
- 新增 `src/apply_bucket_stats_to_all_sources.py`
- 读取：
  - `data/train_test_merged.csv`
  - `outputs/count_quantiles_train_year_month_workingday_hour.csv`
- 对主表衍生 `year` 和 `month` 后，按键 `year,month,workingday,hour` 左连接统计表。
- 只新增列，不覆盖原列：
  - `stat_count_mean_y_m_w_h`
  - `stat_count_std_y_m_w_h`
  - `stat_count_q05_y_m_w_h`
  - `stat_count_q25_y_m_w_h`
  - `stat_count_q50_y_m_w_h`
  - `stat_count_q75_y_m_w_h`
  - `stat_count_q95_y_m_w_h`
- 额外回填 `cv_fold`：
  - 参考文件：`data/train_cv2fold_split.csv`
  - 对齐键：`datetime`
  - 仅 `train/miss_train` 应有值，`test/miss_test` 保持空值。
  - 若参考文件不存在：自动按固定随机种子生成等价 `cv_fold` 映射文件（保证可复现）。
- 输出新文件：`data/train_test_merged_with_bucket_stats.csv`

3. 一致性约束
- 回填目标范围：`source in (train,test,miss_train,miss_test)` 全量样本。
- 主键命中率应为 100%；若个别桶无样本，使用分层兜底（`year+month+workingday` -> `year+workingday+hour` -> 全局均值）补全，确保新增统计列无空值。
- 保持原始行顺序与行数不变。

#### Public Interfaces / Data Contract
- 新增数据文件：`data/train_test_merged_with_bucket_stats.csv`
- 新增统计文件：`outputs/count_quantiles_train_season_month_workingday_hour.csv`
- 原文件 `data/train_test_merged.csv` 不修改。

#### Test Plan
1. 行级一致性
- 新文件行数与原文件一致（17544）。
- `datetime` 顺序与原文件一致。

2. 回填完整性
- 对四类 `source`，新增统计列空值数应为 0。
- 键命中率与兜底后有效覆盖率均为 100%（最终表不得留空）。
- `train/miss_train` 的 `cv_fold` 空值数为 0，`test/miss_test` 的 `cv_fold` 允许为空。

3. 数值合理性
- 每行满足 `q05 <= q25 <= q50 <= q75 <= q95`。
- `stat_count_std_y_m_w_h >= 0`。

4. 变更边界
- 原始核心列（如 `count/source/season/...`）值不被改写，仅新增列。

#### Assumptions
- “回填”解释为新增桶统计特征列，而非覆盖 `count`。
- 输出采用新文件策略，便于与原表做 A/B 对比与回滚。
- `cv_fold` 对齐使用 `datetime` 的一对一键映射，参考文件不存在重复键。
</proposed_plan>
