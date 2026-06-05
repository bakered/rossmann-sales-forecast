# %% [markdown]
# # Models
#
# Progression:
# 1. RMSPE metric
# 2. Single-store log-linear OLS (interpretable baseline)
# 3. Seasonal naive baseline (benchmark to beat)
# 4. XGBoost global model (all stores)

# %% Imports
import os
from pathlib import Path
import pandas as pd
import numpy as np

os.chdir(Path(__file__).resolve().parent.parent)

# %% [markdown]
# ## RMSPE
#
# Official Kaggle metric for this competition.
# Excludes days where actual sales = 0 (closed days) to avoid division by zero.
# Lower is better. Penalises percentage errors equally across small and large stores.

# %%
def rmspe(y_true, y_pred):
    """
    Compute Root Mean Squared Percentage Error.

    Parameters
    ----------
    y_true : array-like of float
        Actual sales. Must contain no NaNs.
    y_pred : array-like of float
        Predicted sales. Must be same length as y_true.

    Returns
    -------
    float
        RMSPE value. Lower is better.

    Assumptions
    -----------
    Rows where y_true == 0 are excluded, consistent with the Kaggle definition.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

# Quick sanity check
assert abs(rmspe([100, 200], [110, 180]) - 0.1) < 0.001
print("RMSPE function OK")

# %% Load data
def load(name):
    p = Path(f"data/processed/{name}")
    if p.with_suffix(".parquet").exists():
        return pd.read_parquet(p.with_suffix(".parquet"))
    return pd.read_csv(p.with_suffix(".csv"), parse_dates=["Date"])

train = load("train_features")
val   = load("val_features")

# Open days only — closed days have Sales=0 and are excluded from RMSPE anyway
train = train[train["Open"] == 1].copy()
val   = val[val["Open"] == 1].copy()

print(f"Train: {len(train):,} rows  |  {train['Date'].min().date()} → {train['Date'].max().date()}")
print(f"Val:   {len(val):,} rows  |  {val['Date'].min().date()} → {val['Date'].max().date()}")
