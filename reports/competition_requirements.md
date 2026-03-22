# Bike Sharing Demand: Submission and Evaluation Requirements

## 1) Data Download Status

- Official Kaggle CLI download attempted with command:
  - `python3 -m kaggle.cli competitions download -c bike-sharing-demand -p data`
- Result: blocked by authentication (`You must authenticate before you can call the Kaggle API.`).
- Current local data source:
  - `data/train.csv`
  - `data/test.csv`
  - `data/sampleSubmission.csv`
  These were fetched from a public mirror to unblock development.

## 2) Officially Confirmed (from Kaggle pages)

From Kaggle Data page for this competition:

- Competition data has 3 csv files: `train.csv`, `test.csv`, `sampleSubmission.csv`.
- Task is hourly demand forecasting.
- Data split statement: train uses earlier days of each month and test uses later days.

## 3) Submission Output (Project Standard)

To ensure direct submit readiness, the project submission file standard is:

- File name: `submission.csv`
- Required columns:
  - `datetime`
  - `count`
- Row count must equal `test.csv` row count (6493 rows).
- Constraints:
  - `count` must be numeric and non-negative.

A ready-to-fill template has been generated:

- `outputs/submission_template.csv`
- `data/sampleSubmission_expected.csv`

Note: current mirror `data/sampleSubmission.csv` contains only `count` column, which is not reliable as official template. Use `outputs/submission_template.csv` or `data/sampleSubmission_expected.csv` for direct submission format (`datetime,count`).

## 4) Evaluation Metric

This competition is evaluated by **RMSLE** (Root Mean Squared Logarithmic Error), i.e. lower is better.
Common definition used by this competition:

`RMSLE = sqrt( mean( (log(pred + 1) - log(actual + 1))^2 ) )`

Note: Kaggle's current unauthenticated pages do not expose the dedicated evaluation panel text in this environment; the RMSLE definition above is inferred from this competition's standard setup and leaderboard score range.

Implementation notes for this project:

- Train on `log1p(count)` is allowed/recommended.
- Final predictions must be transformed back with `expm1` and clipped at `>= 0`.

## 5) Next Step to Switch to Fully Official Data

To replace mirror data with Kaggle-official files, add Kaggle API credentials:

1. Download API token from Kaggle account settings (`kaggle.json`).
2. Put it at `~/.kaggle/kaggle.json`.
3. Run:
  - `chmod 600 ~/.kaggle/kaggle.json`
  - `python3 -m kaggle.cli competitions download -c bike-sharing-demand -p data`
  - `unzip -o data/bike-sharing-demand.zip -d data`

## Sources

- Kaggle competition data page: [https://www.kaggle.com/competitions/bike-sharing-demand/data](https://www.kaggle.com/competitions/bike-sharing-demand/data)
- Kaggle competition overview: [https://www.kaggle.com/competitions/bike-sharing-demand/overview](https://www.kaggle.com/competitions/bike-sharing-demand/overview)
- Kaggle competition leaderboard: [https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard](https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard)
- Kaggle competition rules: [https://www.kaggle.com/competitions/bike-sharing-demand/rules](https://www.kaggle.com/competitions/bike-sharing-demand/rules)
- Kaggle CLI authentication docs: [https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication)
- Public mirror used for temporary local unblock: [https://github.com/sinhabishal77/Kaggle-Bike-Sharing-Demand](https://github.com/sinhabishal77/Kaggle-Bike-Sharing-Demand)

