<proposed_plan>
### 生成带 `source` 标记的合并文件计划

#### Summary
在原合并流程基础上，为每条样本新增 `source` 列，用于标记来源：`train` 或 `test`。最终输出文件仍为 [train_test_merged.csv](/Users/wutenghu/Pycharm/kaggle_bike_sharing_demand/data/train_test_merged.csv)，并按 `datetime` 升序排序。

#### Implementation Changes
1. 读取 [train.csv](/Users/wutenghu/Pycharm/kaggle_bike_sharing_demand/data/train.csv) 与 [test.csv](/Users/wutenghu/Pycharm/kaggle_bike_sharing_demand/data/test.csv)。
2. 在合并前添加来源列：
   - `train` 数据新增 `source='train'`
   - `test` 数据新增 `source='test'`
3. 按行拼接（列并集，保留全部字段）；`test` 的 `casual/registered/count` 保持空值。
4. 将 `datetime` 解析为时间类型并做升序排序。
5. 导出到 `data/train_test_merged.csv`（不写索引）。
6. 执行后记录基础体量（用于后续计划对齐）：
   - 预期行数：`10886 + 6493 = 17379`
   - `source` 分布：`train=10886, test=6493`

#### Public Interfaces / Data Contract
- 输出文件：`data/train_test_merged.csv`
- 新增字段：`source`（字符串枚举：`train` / `test`）
- 其余字段：保持 train/test 列并集
- 排序规则：`datetime` 从小到大

#### Test Plan
1. 行数校验：`rows_out = rows_train + rows_test`。
2. `source` 校验：仅包含 `train`、`test` 两个值，且计数分别等于原始 train/test 行数。
3. 排序校验：`datetime` 单调非降序。
4. 列校验：包含原有列 + `source`。
5. 缺失值校验：`source='test'` 的 `casual/registered/count` 为空。
6. 顺序校验：导出的首尾时间应分别等于 train/test 合并后的全局最小/最大 `datetime`。

#### Assumptions
- `source` 列名当前未与现有列冲突。
- `datetime` 可正常解析，重复时间戳允许存在。
- 此步骤仅生成基础合并表，不在此阶段补齐缺失小时。
</proposed_plan>
