# Freight Rate Prediction

Predicts the posted rate of freight loads. Built for the Spotter Machine Learning Engineer assessment.

## Approach in brief

- **Validation:** time-based split. Train on Jan to Aug 2025, test on Sep to Oct 2025 (the real validation set is Nov to Dec, so a random split would leak future information).
- **Cleaning:** negative weights converted to absolute values; training rows with implausible rate per mile (outside $0.80 to $6.00) dropped; missing values left to the model.
- **Model:** scikit-learn `HistGradientBoostingRegressor` predicting `log(rate per mile)`; prediction is `exp(output) * distance`. City names are not used as features (8 validation cities never appear in training); coordinates are used instead.
- **Results on the Sep to Oct holdout:**

| Model | MAE | MAPE |
|---|---|---|
| Baseline: distance x equipment rate per mile | $229 | 10.5% |
| Gradient boosting (final) | $131 | 5.6% |

Details and the December chart are in `Freight_Rate_Report.docx`.

## Repository layout

```
.
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
├── 01_explore.ipynb            # exploration and experiments
├── train_and_predict.py        # full pipeline: validate, train, predict
├── score.py                    # scorer provided by Spotter
├── requirements.txt
├── validation_predictions.csv  # output: load_id,predicted_rate
├── scorer_results/
│   └── candidate_december.png  # output: December chart
├── Freight_Rate_Report.docx
└── README.md
```

> The data files are provided by Spotter. Place them in `data/` with the names above (underscores, not hyphens) before running.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Run

```bash
python train_and_predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`train_and_predict.py` will:

1. Run the time-based validation and print the baseline and model scores.
2. Retrain on all labelled data (Jan to Oct).
3. Write `validation_predictions.csv` (12,000 rows, `load_id,predicted_rate`).
4. Fill the `predicted_rate` column of `data/december_chart_inputs.csv`.

`score.py` validates both files and creates `scorer_results/candidate_december.png`.

Training takes well under a minute on a normal laptop. Results are seeded (`random_state=42`), though tiny differences can appear across library versions.

## December chart inputs

`december_chart_inputs.csv` has no `market_index` or `quote_signal`, so the script uses each day's average from `validation.csv` (which covers every December day) and the lane coordinates from the training data.

## Notebook

`01_explore.ipynb` contains the data exploration, the baselines, and the city-name experiment that led to the final feature set.
