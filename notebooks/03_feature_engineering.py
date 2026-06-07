# %% [markdown]
# # Feature Engineering
#
# Builds the modelling dataset from raw train.csv + store.csv.
# Output saved to data/processed/features.parquet.
#
# All lag and rolling features are computed on the full dataset sorted by date,
# so test-period rows naturally look back into training-period actuals — no leakage.
# The train/test split is applied at the end; it must NOT be applied before computing lags.

# %% Imports
import os
from pathlib import Path
import pandas as pd
import numpy as np

os.chdir(Path(__file__).resolve().parent.parent)

Path("data/processed").mkdir(parents=True, exist_ok=True)

# %% Load raw data
train = pd.read_csv("data/raw/train.csv", parse_dates=["Date"],
                    dtype={"StateHoliday": str})
store = pd.read_csv("data/raw/store.csv")

df = train.merge(store, on="Store", how="left")
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

print(f"Rows: {len(df):,}  |  Stores: {df['Store'].nunique()}")

# %% [markdown]
# ## 1. Date features

# %%
df["Year"]        = df["Date"].dt.year
df["Month"]       = df["Date"].dt.month
df["WeekOfYear"]  = df["Date"].dt.isocalendar().week.astype(int)
df["DayOfMonth"]  = df["Date"].dt.day
# DayOfWeek already in raw data (1=Mon … 7=Sun)

# %% [markdown]
# ## 2. Christmas / New Year dummies
#
# One binary column per calendar date for the high-variance holiday window.
# Model learns each day's multiplier independently.

# %%
md = df["Date"].dt.month * 100 + df["Date"].dt.day   # MMDD integer for fast comparison

# Dec 15–24 (pre-Christmas ramp)
for day in range(15, 25):
    df[f"is_dec_{day:02d}"] = (md == 1200 + day).astype(int)

# Dec 26–31 (post-Christmas run-down into New Year)
for day in range(26, 32):
    df[f"is_dec_{day:02d}"] = (md == 1200 + day).astype(int)

# Jan 2 — first normal trading day of new year (Jan 1 is a public holiday)
df["is_jan_02"] = (md == 102).astype(int)

xmas_cols = [c for c in df.columns if c.startswith("is_dec_") or c == "is_jan_02"]
print(f"Holiday dummy columns: {xmas_cols}")

# %% [markdown]
# ## 3. Other calendar flags

# %%
# is_pre_holiday: day immediately before a state or public holiday
# Flags stockpiling behaviour on the last open day before a closure.
df["StateHoliday_next"] = df.groupby("Store")["StateHoliday"].shift(-1).fillna("0")
df["is_pre_holiday"] = (df["StateHoliday_next"] != "0").astype(int)
df.drop(columns="StateHoliday_next", inplace=True)

# is_bridge_day: weekday sandwiched between a public holiday and a weekend.
# e.g. Thursday holiday → Friday is a bridge day.
df["StateHoliday_prev"] = df.groupby("Store")["StateHoliday"].shift(1).fillna("0")
df["StateHoliday_next"] = df.groupby("Store")["StateHoliday"].shift(-1).fillna("0")
is_weekday    = df["DayOfWeek"].between(1, 5)
prev_holiday  = df["StateHoliday_prev"] != "0"
next_holiday  = df["StateHoliday_next"] != "0"
prev_weekend  = df["DayOfWeek"].shift(1).fillna(0).isin([6, 7])
next_weekend  = df["DayOfWeek"].shift(-1).fillna(0).isin([6, 7])
df["is_bridge_day"] = (
    is_weekday & ((prev_holiday & next_weekend) | (prev_weekend & next_holiday))
).astype(int)
df.drop(columns=["StateHoliday_prev", "StateHoliday_next"], inplace=True)

# is_month_end: last 3 calendar days of the month (payday spending pattern)
df["is_month_end"] = (df["DayOfMonth"] >= df["Date"].dt.days_in_month - 2).astype(int)

print("Calendar flags created.")
print(f"Pre-holiday days:  {df['is_pre_holiday'].sum():,}")
print(f"Bridge days:       {df['is_bridge_day'].sum():,}")
print(f"Month-end days:    {df['is_month_end'].sum():,}")

# %% [markdown]
# ## 4. Promo2_active
#
# A store's Promo2 is active on a given day if (a) the store participates in Promo2
# and (b) that calendar month appears in its PromoInterval.

# %%
MONTH_MAP = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12}

def is_promo2_active(promo2, interval, month):
    """
    Return 1 if Promo2 is active for this store in this month, else 0.

    Parameters
    ----------
    promo2   : int — store-level participation flag
    interval : str — PromoInterval e.g. "Jan,Apr,Jul,Oct", or NaN
    month    : int — calendar month (1–12)

    Returns
    -------
    int
    """
    if promo2 == 0 or not isinstance(interval, str):
        return 0
    return int(month in [MONTH_MAP[m.strip()] for m in interval.split(",")])

df["Promo2_active"] = df.apply(
    lambda r: is_promo2_active(r["Promo2"], r["PromoInterval"], r["Month"]), axis=1
)
print(f"Promo2_active rate: {df['Promo2_active'].mean():.1%}")

# %% [markdown]
# ## 5. Competition features

# %%
# CompetitionDistance: ~0.3% NaN — likely stores with no known nearby competitor.
# Impute with max observed distance and add a binary flag.
max_dist = df["CompetitionDistance"].max()
df["no_competitor"]      = df["CompetitionDistance"].isna().astype(int)
df["CompetitionDistance"] = df["CompetitionDistance"].fillna(max_dist)

# months_since_competitor_opened: how long the nearest competitor has been trading.
# NaN means competitor predates the sample or there is no competitor — set to large value.
comp_open = pd.to_datetime(
    df["CompetitionOpenSinceYear"].astype("Int64").astype(str) + "-" +
    df["CompetitionOpenSinceMonth"].astype("Int64").astype(str) + "-01",
    errors="coerce"
)
months_since = ((df["Date"].dt.year  - comp_open.dt.year) * 12 +
                (df["Date"].dt.month - comp_open.dt.month))
# Negative values mean competitor hadn't opened yet on that date
months_since = months_since.clip(lower=0)
# NaN (no opening date recorded) → 999 (competitor long-established or absent)
df["months_since_competitor_opened"] = months_since.fillna(999).astype(int)

# New-competition flags — signal that lag features are unreliable.
# When a competitor opens, the same-condition lag (14 days prior) still reflects
# pre-competition sales. The model needs to know to discount the lag during
# this adjustment window.
#
# comp_opened_last_6m: binary — competitor opened within the last 6 months.
#   Triggers during the adjustment period; clean 0/1 split for tree models.
# days_since_comp_opened: continuous 0–180 — lets the model learn that the
#   impact fades as the lag window fills with post-competition observations.
#   0 for all rows where competition is long-established or absent.
days_since_comp = (df["Date"] - comp_open).dt.days.clip(lower=0)
df["comp_opened_last_6m"]   = ((days_since_comp >= 0) & (days_since_comp <= 180)).astype(int)
df["days_since_comp_opened"] = days_since_comp.where(days_since_comp <= 180, 0).fillna(0).astype(int)

print("Competition features created.")
print(f"No-competitor stores:           {df['no_competitor'].sum():,} rows")
print(f"comp_opened_last_6m rows:       {df['comp_opened_last_6m'].sum():,}")
print(f"days_since_comp_opened > 0:     {(df['days_since_comp_opened'] > 0).sum():,}")

# %% [markdown]
# ## 6. Lag and rolling features
#
# All computed per store, sorted by date, on open days only.
# Closed days are excluded from lag computations — a store's sales history
# should not include zeros from days it was shut.
#
# same_cond = same DayOfWeek AND same Promo status.
# This is the most comparable prior observation: same weekly timing, same promotional environment.
# Median gap between same_cond occurrences is 14 days (the alternating weekly schedule).

# %%
# Work on open days only for lag computation, then merge back
open_df = df[df["Open"] == 1].copy()
open_df = open_df.sort_values(["Store", "Date"])

# --- Standard lags ---
open_df["sales_lag_7"]  = open_df.groupby("Store")["Sales"].shift(7)
open_df["sales_lag_14"] = open_df.groupby("Store")["Sales"].shift(14)

# --- Same-condition lags and rolling ---
# Group by (Store, DayOfWeek, Promo) — each group contains only days with the
# same weekday and promo status for a given store.
same_cond = open_df.groupby(["Store", "DayOfWeek", "Promo"])

open_df["sales_lag_same_cond"]   = same_cond["Sales"].shift(1)
open_df["sales_roll4_same_cond"] = same_cond["Sales"].shift(1).rolling(4, min_periods=2).mean().values
open_df["sales_roll8_same_cond"] = same_cond["Sales"].shift(1).rolling(8, min_periods=4).mean().values

# TO REVIEW: rolling() after groupby+shift operates on the pre-grouped series.
# Verify that values align correctly with the original index after reset.

# --- Store-level trend: 56-day rolling mean (general level, not same-cond) ---
open_df["store_trend_56"] = (
    open_df.groupby("Store")["Sales"]
    .transform(lambda s: s.shift(1).rolling(56, min_periods=14).mean())
)

# Merge lag features back onto full df (closed days get NaN, which is correct)
lag_cols = ["sales_lag_7", "sales_lag_14", "sales_lag_same_cond",
            "sales_roll4_same_cond", "sales_roll8_same_cond", "store_trend_56"]
df = df.merge(open_df[["Store", "Date"] + lag_cols], on=["Store", "Date"], how="left")

print("Lag features created.")
print(df[lag_cols].notna().mean().round(3).to_string())

# %% [markdown]
# ## 7. Splits
#
# Three splits, all time-based:
#
# | Split | Period | Purpose |
# |---|---|---|
# | train | 2013-01-01 → 2015-04-30 | Model fitting |
# | val   | 2015-05-01 → 2015-07-31 | Evaluation during development |
# | trainval | 2013-01-01 → 2015-07-31 | Full data for final model before Kaggle submission |
#
# Splits are applied AFTER feature engineering so lag features for validation
# correctly look back into training actuals — no leakage.

# %%
TRAIN_END   = "2015-04-30"
VAL_START   = "2015-05-01"

train_df    = df[df["Date"] <= TRAIN_END].copy()
val_df      = df[df["Date"] >= VAL_START].copy()
trainval_df = df.copy()

print(f"Train:    {train_df['Date'].min().date()} → {train_df['Date'].max().date()}  ({len(train_df):,} rows)")
print(f"Val:      {val_df['Date'].min().date()}  → {val_df['Date'].max().date()}   ({len(val_df):,} rows)")
print(f"Trainval: {trainval_df['Date'].min().date()} → {trainval_df['Date'].max().date()}  ({len(trainval_df):,} rows)")

# %% Save
def save(df_, name):
    try:
        df_.to_parquet(f"data/processed/{name}.parquet", index=False)
        print(f"  {name}.parquet")
    except ImportError:
        df_.to_csv(f"data/processed/{name}.csv", index=False)
        print(f"  {name}.csv  (install pyarrow for parquet)")

print("\nSaved:")
save(train_df,    "train_features")
save(val_df,      "val_features")
save(trainval_df, "trainval_features")
print(f"Features: {[c for c in df.columns if c not in train.columns and c not in ['Year','Month']]}")
