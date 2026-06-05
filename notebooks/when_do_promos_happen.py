# %% [markdown]
# # When Do Promotions Happen?
#
# Explores the timing and distribution of Promo (daily promotion) and Promo2
# (recurring scheme) across stores, days of week, and months.
#
# Promo2_active is engineered: a store's Promo2 counts as active on a given day
# if (a) the store participates in Promo2, and (b) that month appears in its
# PromoInterval (e.g. "Jan,Apr,Jul,Oct").

# %% Imports
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.chdir(Path(__file__).resolve().parent.parent)

# %% Load and merge
train = pd.read_csv("data/raw/train.csv", parse_dates=["Date"],
                    dtype={"StateHoliday": str})
store = pd.read_csv("data/raw/store.csv")
df = train.merge(store, on="Store", how="left")

# Open days only — closed days always have Promo=0 and are uninformative
df = df[df["Open"] == 1].copy()

df["Month"]     = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek + 1  # 1=Mon … 7=Sun

# %% Engineer Promo2_active
# Map abbreviated month names to integers
MONTH_MAP = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12}

def is_promo2_active(promo2, interval, month):
    """
    Return 1 if Promo2 is running in this month for this store, else 0.

    Parameters
    ----------
    promo2   : int   — store-level participation flag (0 or 1)
    interval : str   — PromoInterval string e.g. "Jan,Apr,Jul,Oct", or NaN
    month    : int   — calendar month (1–12)

    Returns
    -------
    int : 1 if active, 0 otherwise
    """
    if promo2 == 0 or not isinstance(interval, str):
        return 0
    active_months = [MONTH_MAP[m.strip()] for m in interval.split(",")]
    return int(month in active_months)

df["Promo2_active"] = df.apply(
    lambda r: is_promo2_active(r["Promo2"], r["PromoInterval"], r["Month"]), axis=1
)

print(f"Promo    active on {df['Promo'].mean():.1%} of open days")
print(f"Promo2   active on {df['Promo2_active'].mean():.1%} of open days")

# %% [markdown]
# ## Promo frequency per store

# %% Per-store promo rates
store_promo  = df.groupby("Store")["Promo"].mean().reset_index(name="promo_rate")
store_promo2 = df.groupby("Store")["Promo2_active"].mean().reset_index(name="promo2_rate")

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["Promo — fraction of open days per store",
                                    "Promo2 active — fraction of open days per store"],
                    horizontal_spacing=0.12)

fig.add_trace(go.Histogram(x=store_promo["promo_rate"], nbinsx=20,
                           marker_color="steelblue", name="Promo"), row=1, col=1)
fig.add_trace(go.Histogram(x=store_promo2["promo2_rate"], nbinsx=20,
                           marker_color="coral", name="Promo2"), row=1, col=2)

fig.update_xaxes(title_text="Fraction of open days", row=1, col=1)
fig.update_xaxes(title_text="Fraction of open days", row=1, col=2)
fig.update_yaxes(title_text="Number of stores", row=1, col=1)
fig.update_layout(title_text="Promotion frequency per store",
                  showlegend=False, template="plotly_white", height=450, width=1000)
fig.show()

# %% [markdown]
# ## PromoStores — stores with Promo on more than 40% of open days

# %% Define PromoStores and merge flag onto main df
# Stores running Promo on >40% of open days are classified as PromoStores.
# This threshold captures the natural split visible in the per-store histogram above.
store_promo["PromoStore"] = (store_promo["promo_rate"] > 0.4).astype(int)

df = df.merge(store_promo[["Store", "PromoStore"]], on="Store", how="left")

n_promo    = store_promo["PromoStore"].sum()
n_nonpromo = len(store_promo) - n_promo
print(f"PromoStores (rate > 40%): {n_promo} stores")
print(f"Non-PromoStores:          {n_nonpromo} stores")

# %% Sales distribution — PromoStores vs non-PromoStores
sales_promo    = df[df["PromoStore"] == 1]["Sales"]
sales_nonpromo = df[df["PromoStore"] == 0]["Sales"]

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["Daily sales distribution",
                                    "Mean sales by day of week"],
                    horizontal_spacing=0.12)

# Overlapping histograms
fig.add_trace(go.Histogram(x=sales_promo, nbinsx=60, name="PromoStore",
                           marker_color="steelblue", opacity=0.65,
                           histnorm="probability density"), row=1, col=1)
fig.add_trace(go.Histogram(x=sales_nonpromo, nbinsx=60, name="Non-PromoStore",
                           marker_color="coral", opacity=0.65,
                           histnorm="probability density"), row=1, col=1)

# Mean sales by day of week
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_promo    = df[df["PromoStore"] == 1].groupby("DayOfWeek")["Sales"].mean()
dow_nonpromo = df[df["PromoStore"] == 0].groupby("DayOfWeek")["Sales"].mean()

fig.add_trace(go.Bar(name="PromoStore", x=dow_labels, y=dow_promo.values,
                     marker_color="steelblue", opacity=0.85, showlegend=False), row=1, col=2)
fig.add_trace(go.Bar(name="Non-PromoStore", x=dow_labels, y=dow_nonpromo.values,
                     marker_color="coral", opacity=0.85, showlegend=False), row=1, col=2)

fig.update_xaxes(title_text="Daily sales (€)", row=1, col=1)
fig.update_yaxes(title_text="Density", row=1, col=1)
fig.update_yaxes(title_text="Mean daily sales (€)", row=1, col=2)
fig.update_layout(barmode="group", template="plotly_white", height=450, width=1100,
                  title_text="PromoStores vs Non-PromoStores — sales",
                  legend=dict(x=0.38, y=1.12, orientation="h"))
fig.show()

# %% Summary stats — PromoStores vs non-PromoStores
summary = df.groupby("PromoStore")["Sales"].agg(
    mean="mean", median="median", std="std", p25=lambda x: x.quantile(0.25),
    p75=lambda x: x.quantile(0.75)
).round(0)
summary.index = ["Non-PromoStore", "PromoStore"]
print(summary)

lift = summary.loc["PromoStore", "mean"] - summary.loc["Non-PromoStore", "mean"]
lift_pct = lift / summary.loc["Non-PromoStore", "mean"] * 100
print(f"\nMean sales lift (PromoStore vs Non-PromoStore): +€{lift:,.0f} (+{lift_pct:.1f}%)")
# TO REVIEW: this comparison conflates the promo effect with store characteristics —
# PromoStores may differ from non-PromoStores in size, type, location etc.
# Phase 3 propensity analysis addresses this more carefully at the daily level.

# %% [markdown]
# ## Promo frequency by day of week

# %% By day of week
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

promo_dow  = df.groupby("DayOfWeek")["Promo"].mean()
promo2_dow = df.groupby("DayOfWeek")["Promo2_active"].mean()

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["Promo — by day of week",
                                    "Promo2 active — by day of week"])

fig.add_trace(go.Bar(x=dow_labels, y=promo_dow.values,
                     marker_color="steelblue", name="Promo"), row=1, col=1)
fig.add_trace(go.Bar(x=dow_labels, y=promo2_dow.values,
                     marker_color="coral", name="Promo2"), row=1, col=2)

fig.update_yaxes(title_text="Fraction of open days", tickformat=".0%", row=1, col=1)
fig.update_yaxes(title_text="Fraction of open days", tickformat=".0%", row=1, col=2)
fig.update_layout(title_text="Promotion frequency by day of week",
                  showlegend=False, template="plotly_white", height=400)
fig.show()

# %% [markdown]
# ## Promo frequency by month

# %% By month
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

promo_month  = df.groupby("Month")["Promo"].mean()
promo2_month = df.groupby("Month")["Promo2_active"].mean()

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["Promo — by month",
                                    "Promo2 active — by month"])

fig.add_trace(go.Bar(x=month_labels, y=promo_month.values,
                     marker_color="steelblue", name="Promo"), row=1, col=1)
fig.add_trace(go.Bar(x=month_labels, y=promo2_month.values,
                     marker_color="coral", name="Promo2"), row=1, col=2)

fig.update_yaxes(title_text="Fraction of open days", tickformat=".0%", row=1, col=1)
fig.update_yaxes(title_text="Fraction of open days", tickformat=".0%", row=1, col=2)
fig.update_layout(title_text="Promotion frequency by month",
                  showlegend=False, template="plotly_white", height=400)
fig.show()

# %% [markdown]
# ## Overlay — Promo vs Promo2 on the same axes

# %% Overlay by day of week
fig = go.Figure()
fig.add_trace(go.Bar(name="Promo", x=dow_labels, y=promo_dow.values,
                     marker_color="steelblue", opacity=0.85))
fig.add_trace(go.Bar(name="Promo2 active", x=dow_labels, y=promo2_dow.values,
                     marker_color="coral", opacity=0.85))
fig.update_layout(barmode="group", template="plotly_white",
                  title="Promo vs Promo2 — by day of week",
                  yaxis=dict(title="Fraction of open days", tickformat=".0%"),
                  height=400)
fig.show()

# %% Overlay by month
fig = go.Figure()
fig.add_trace(go.Bar(name="Promo", x=month_labels, y=promo_month.values,
                     marker_color="steelblue", opacity=0.85))
fig.add_trace(go.Bar(name="Promo2 active", x=month_labels, y=promo2_month.values,
                     marker_color="coral", opacity=0.85))
fig.update_layout(barmode="group", template="plotly_white",
                  title="Promo vs Promo2 — by month",
                  yaxis=dict(title="Fraction of open days", tickformat=".0%"),
                  height=400)
fig.show()

# %% [markdown]
# ## Sales over time — 10 random stores, coloured by day type
#
# Each bar is one day. Colours:
# - **Steelblue** — weekday with Promo=1
# - **Coral** — weekday with Promo=0
# - **Lightgrey** — weekend (Saturday/Sunday)
#
# Promos only occur on weekdays so weekends are always non-promo.
# Use Plotly's zoom to inspect individual periods.

# %% Pick 10 random stores (fixed seed for reproducibility)
import numpy as np
rng = np.random.default_rng(42)
sample_stores = sorted(rng.choice(df["Store"].unique(), size=10, replace=False))
print("Sampled stores:", sample_stores)

# %% Build colour column
# Reload full df including closed days so the time axis is continuous,
# then filter to the sample stores.
train_full = pd.read_csv("data/raw/train.csv", parse_dates=["Date"],
                         dtype={"StateHoliday": str})

sample_df = train_full[train_full["Store"].isin(sample_stores)].copy()
sample_df["DayOfWeek"] = sample_df["Date"].dt.dayofweek + 1  # 1=Mon … 7=Sun
sample_df["is_weekend"] = sample_df["DayOfWeek"] >= 6

def day_colour(row):
    """Assign display colour based on weekend/promo status."""
    if row["is_weekend"]:
        return "lightgrey"
    return "steelblue" if row["Promo"] == 1 else "coral"

sample_df["colour"] = sample_df.apply(day_colour, axis=1)

# %% Plot — 5 rows x 2 cols, one subplot per store
fig = make_subplots(rows=5, cols=2,
                    subplot_titles=[f"Store {s}" for s in sample_stores],
                    shared_xaxes=False,
                    vertical_spacing=0.07,
                    horizontal_spacing=0.08)

for i, store_id in enumerate(sample_stores):
    row = i // 2 + 1
    col = i % 2 + 1
    sdf = sample_df[sample_df["Store"] == store_id].sort_values("Date")

    fig.add_trace(
        go.Bar(
            x=sdf["Date"],
            y=sdf["Sales"],
            marker_color=sdf["colour"],
            showlegend=(i == 0),  # only add legend entries once
            name="temp",          # overridden by legend traces below
        ),
        row=row, col=col
    )
    fig.update_yaxes(title_text="Sales (€)", row=row, col=col)

# Add invisible legend traces for the three categories
for label, colour in [("Weekend", "lightgrey"),
                      ("Weekday — Promo", "steelblue"),
                      ("Weekday — No promo", "coral")]:
    fig.add_trace(go.Bar(x=[None], y=[None], name=label,
                         marker_color=colour, showlegend=True))

fig.update_layout(
    title_text="Daily sales for 10 random stores — coloured by day type",
    template="plotly_white",
    height=1400,
    width=1200,
    legend=dict(orientation="h", x=0.25, y=1.01),
    bargap=0.1,
)
fig.show()
