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
                    subplot_titles=["Promo — rate per store",
                                    "Promo2 active — rate per store"])

fig.add_trace(go.Histogram(x=store_promo["promo_rate"], nbinsx=40,
                           marker_color="steelblue", name="Promo"), row=1, col=1)
fig.add_trace(go.Histogram(x=store_promo2["promo2_rate"], nbinsx=40,
                           marker_color="coral", name="Promo2"), row=1, col=2)

fig.update_xaxes(title_text="Fraction of open days with promotion", row=1, col=1)
fig.update_xaxes(title_text="Fraction of open days with promotion", row=1, col=2)
fig.update_yaxes(title_text="Number of stores", row=1, col=1)
fig.update_layout(title_text="Promotion frequency per store",
                  showlegend=False, template="plotly_white", height=400)
fig.show()

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
