# Rossmann Sales Forecasting & Promo Analysis

Time-series forecasting and causal inference on the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset (Kaggle).

## Project aim

Two objectives:

1. **Forecasting** — benchmark multiple models (naive baselines, Holt-Winters, SARIMA, XGBoost, LightGBM) on daily store sales for 1,115 German Rossmann drugstores. Models are evaluated on RMSPE (the Kaggle competition metric) on a held-out test set mirroring the competition holdout period (July–September 2015).

2. **Promo causal analysis** — move beyond a naive "promo lifts sales" claim. Promotions are endogenous (planned based on expected demand), so a naive comparison of promo vs non-promo sales days is biased. A propensity score model is used to estimate the causal effect of promotions via Inverse Probability Weighting (IPW), and the naive and adjusted estimates are compared.

## Data

The raw Rossmann data is not committed to this repository (Kaggle terms of use). See [`data/README.md`](data/README.md) for download instructions.

## Repository structure

```
data/
  raw/           # Kaggle CSVs (gitignored)
  processed/     # Engineered features (gitignored)
  README.md      # Download instructions
notebooks/
  01_eda.ipynb
  02_forecasting.ipynb
  03_promo.ipynb
src/             # Reusable functions extracted from notebooks
outputs/         # Saved plots and result tables
requirements.txt
```

## Reproducing results

```bash
pip install -r requirements.txt
# Download data per data/README.md, then run notebooks in order
```
