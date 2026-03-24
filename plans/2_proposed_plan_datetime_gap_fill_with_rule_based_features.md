<proposed_plan>
### 按日期规则补齐缺失时间并重建关键特征（含 `hour`）

#### Summary
对 [train_test_merged.csv](/Users/wutenghu/Pycharm/kaggle_bike_sharing_demand/data/train_test_merged.csv) 做小时级缺失补齐，保证 `datetime` 连续；新增缺失行 `source` 为 `miss_train` 或 `miss_test`。  
你指定的规则会被固化为：`workingday/season/holiday` 由日期重算，新增 `hour` 特征；`weather,temp,atemp,humidity,windspeed` 用时间线性插值补齐；并保证插值后导出的数值文本格式与原始列格式一致。

#### Implementation Changes
1. 读取 `train.csv`、`test.csv`、`train_test_merged.csv`，统一 `datetime` 为小时时间戳并排序。
2. 识别 merged 在 `[min(datetime), max(datetime)]` 内的缺失小时点，逐点补 1 行：
   - 在 `train` 日期范围内优先标记 `source=miss_train`；
   - 否则在 `test` 日期范围内标记 `source=miss_test`；
   - 重叠区不补双行（单行、train 优先）。
3. 对补齐行按日期重建离散字段：
   - `hour = datetime.hour`
   - `season` 根据数据提供规则做映射
   - `holiday` 根据数据提供规则做映射
   - `workingday` 根据数据提供的规则做映射
4. 对连续数值特征 `weather,temp,atemp,humidity,windspeed`：
   - 按完整时间序列做线性插值（time-based）；
   - 头尾再 `ffill/bfill`；
   - `weather` 最终四舍五入并裁剪到合法类别区间（1-4）。
5. 标签列策略：
   - `miss_test` 的 `casual,registered,count` 保持空值；
   - `miss_train` 的 `casual,registered,count` 基于时间线性插值填充，最终 `round` 为整数（并保证非负；`count=casual+registered`）。
6. 输出格式约束：
   - 整数列写出为整数文本，不带 `.0`；
   - 浮点列按原始列精度风格写出
7. 按 `datetime` 升序输出回 `data/train_test_merged.csv`。
8. 保证“只改新增行和确需插值列”：原始 `source in (train,test)` 的现有标签值不重算、不覆盖。

#### Public Interfaces / Data Contract
- 更新文件：`data/train_test_merged.csv`
- `source` 扩展为：`train,test,miss_train,miss_test`
- 新增字段：`hour`
- 保证 `datetime` 小时级连续无缺口。

#### Test Plan
1. 连续性：`datetime` 每小时连续，缺失点为 0。
2. 来源标记：新增行仅含 `miss_train/miss_test`，且总新增行数等于缺失小时数。
3. 规则一致性：
   - `season` 与月份映射一致；
   - `workingday == (weekday && !holiday)` 全量成立；
   - `hour == datetime.hour` 全量成立。
4. 插值结果：
   - `weather,temp,atemp,humidity,windspeed` 在补齐行不为空；
   - `weather` 值域在 1-4。
5. 标签约束：
   - `source=miss_test` 的 `casual/registered/count` 全为空；
   - `source=miss_train` 的三列由线性插值生成并 `round` 为整数，且 `count=casual+registered`。
6. 格式约束：
   - 整数列无 `.0`；
   - `temp/atemp/windspeed` 小数位上限分别为 `2/3/4`。
7. 体量校验（当前数据基线）：
   - 最终行数 `17544`
   - `source` 分布 `train=10886, test=6493, miss_train=163, miss_test=2`

#### Assumptions
- 节假日判定与原始 `holiday` 日期一致。
- 仅补齐缺失时间点，不改写原有 `train/test` 行的已存在值。
</proposed_plan>
