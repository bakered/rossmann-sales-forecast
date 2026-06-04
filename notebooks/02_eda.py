# %% [markdown]
# # Rossmann EDA — Polished Analysis
# Covers: data quality, sales distributions, time trends, seasonality,
# store-level variation, promo patterns, and correlations.
# Key charts are saved to outputs/.

# %% Imports and config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (12, 4), "figure.dpi": 120, "axes.titlesize": 11})

# %% [markdown]
# ## 1. Load data

# %% Load
train = pd.read_csv("data/raw/train.csv", parse_dates=["Date"],
                    dtype={"StateHoliday": str})
store = pd.read_csv("data/raw/store.csv")
test  = pd.read_csv("data/raw/test.csv",  parse_dates=["Date"])

# Merge store metadata onto train
df = train.merge(store, on="Store", how="left")

print(f"train : {train.shape[0]:,} rows × {train.shape[1]} cols")
print(f"store : {store.shape[0]:,} rows × {store.shape[1]} cols")
print(f"test  : {test.shape[0]:,} rows  × {test.shape[1]} cols")
print(f"\nDate range : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Stores     : {df['Store'].nunique()}")

# %% [markdown]
# ## 2. Data quality

# %% Dtypes
print(df.dtypes.to_string())

# %% Null counts
nulls = df.isnull().sum()
nulls = nulls[nulls > 0].rename("null_count").to_frame()
nulls["pct"] = (nulls["null_count"] / len(df) * 100).round(2)
print(nulls)
# CompetitionDistance is the main column with NaNs (~0.3% of rows).
# These likely represent stores with no nearby competitor.
# TO REVIEW: imputing with a large value (e.g. max + 1) vs. a separate flag column.

# %% Open=0 rows
closed = df[df["Open"] == 0]
print(f"\nClosed days (Open=0): {len(closed):,} ({len(closed)/len(df):.1%} of all rows)")
print(f"Sales on closed days: min={closed['Sales'].min()}, max={closed['Sales'].max()}")
# All closed days have Sales=0. These are excluded from most analyses below
# to avoid distorting distributions and aggregations.

# %% [markdown]
# ## 3. Sales distribution

# %% Sales histogram — open days only
open_df = df[df["Open"] == 1].copy()

fig, axes = plt.subplots(1, 2)
axes[0].hist(open_df["Sales"], bins=80, color="steelblue", edgecolor="none")
axes[0].set(title="Daily sales (open days)", xlabel="Sales (€)", ylabel="Count")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

axes[1].hist(np.log1p(open_df["Sales"]), bins=80, color="steelblue", edgecolor="none")
axes[1].set(title="log(Sales + 1) — open days", xlabel="log Sales", ylabel="Count")

plt.suptitle("Sales distribution", y=1.01, fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUTS / "01_sales_distribution.png", bbox_inches="tight")
plt.show()

print(f"Sales stats (open days):\n{open_df['Sales'].describe().round(0)}")

# %% [markdown]
# ## 4. Sales over time

# %% Aggregate daily sales and rolling 28-day mean
daily = df.groupby("Date")["Sales"].sum().reset_index()
daily["rolling_28"] = daily["Sales"].rolling(28, center=True).mean()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily["Date"], daily["Sales"], lw=0.6, color="steelblue", alpha=0.6, label="Daily total")
ax.plot(daily["Date"], daily["rolling_28"], lw=1.8, color="navy", label="28-day rolling mean")
ax.set(title="Total daily sales across all stores (2013–2015)",
       xlabel="Date", ylabel="Total sales (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "02_sales_over_time.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Seasonality

# %% Day of week
open_df["DayOfWeek"] = open_df["Date"].dt.dayofweek + 1  # 1=Mon … 7=Sun
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

fig, axes = plt.subplots(1, 2)
dow_sales = open_df.groupby("DayOfWeek")["Sales"].mean()
axes[0].bar(dow_labels, dow_sales.values, color="steelblue", edgecolor="none")
axes[0].set(title="Mean sales by day of week", ylabel="Mean sales (€)")

dow_txn = open_df.groupby("DayOfWeek")["Customers"].mean()
axes[1].bar(dow_labels, dow_txn.values, color="coral", edgecolor="none")
axes[1].set(title="Mean customers by day of week", ylabel="Mean customers")

plt.tight_layout()
plt.savefig(OUTPUTS / "03_seasonality_dow.png", bbox_inches="tight")
plt.show()

# %% Month
open_df["Month"] = open_df["Date"].dt.month
month_sales = open_df.groupby("Month")["Sales"].mean()

fig, ax = plt.subplots()
ax.bar(month_sales.index, month_sales.values, color="steelblue", edgecolor="none")
ax.set(title="Mean sales by month", xlabel="Month", ylabel="Mean sales (€)")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
plt.tight_layout()
plt.savefig(OUTPUTS / "04_seasonality_month.png", bbox_inches="tight")
plt.show()

# %% Week of year
open_df["WeekOfYear"] = open_df["Date"].dt.isocalendar().week.astype(int)
woy_sales = open_df.groupby("WeekOfYear")["Sales"].mean()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(woy_sales.index, woy_sales.values, color="steelblue", lw=1.5, marker="o", markersize=3)
ax.set(title="Mean sales by week of year", xlabel="Week of year", ylabel="Mean sales (€)")
plt.tight_layout()
plt.savefig(OUTPUTS / "05_seasonality_woy.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Store-level variation

# %% Distribution of per-store mean sales
store_stats = open_df.groupby("Store")["Sales"].agg(["mean", "std", "count"])

fig, ax = plt.subplots()
ax.hist(store_stats["mean"], bins=60, color="steelblue", edgecolor="none")
ax.set(title=f"Distribution of mean daily sales across {store_stats.shape[0]} stores",
       xlabel="Mean daily sales (€)", ylabel="Number of stores")
plt.tight_layout()
plt.savefig(OUTPUTS / "06_store_variation.png", bbox_inches="tight")
plt.show()

top5    = store_stats["mean"].nlargest(5)
bottom5 = store_stats["mean"].nsmallest(5)
print("Top 5 stores by mean sales:\n", top5.round(0))
print("\nBottom 5 stores by mean sales:\n", bottom5.round(0))

# %% [markdown]
# ## 7. Store types and assortment

# %% Sales by StoreType and Assortment
fig, axes = plt.subplots(1, 2)

type_sales = open_df.groupby("StoreType")["Sales"].mean()
axes[0].bar(type_sales.index, type_sales.values, color="steelblue", edgecolor="none")
axes[0].set(title="Mean sales by StoreType", ylabel="Mean daily sales (€)")

assort_sales = open_df.groupby("Assortment")["Sales"].mean()
axes[1].bar(assort_sales.index, assort_sales.values, color="coral", edgecolor="none")
axes[1].set(title="Mean sales by Assortment", ylabel="Mean daily sales (€)")

plt.tight_layout()
plt.savefig(OUTPUTS / "07_storetype_assortment.png", bbox_inches="tight")
plt.show()

# Store counts per category
print("Store count by StoreType:\n", df.drop_duplicates("Store")["StoreType"].value_counts().to_string())
print("\nStore count by Assortment:\n", df.drop_duplicates("Store")["Assortment"].value_counts().to_string())

# %% StateHoliday
# StateHoliday values: 0 = none, a = public holiday, b = Easter, c = Christmas
print("\nStateHoliday value counts:")
print(open_df["StateHoliday"].value_counts().to_string())

holiday_sales = open_df.groupby("StateHoliday")["Sales"].mean().rename(
    index={"0": "None", "a": "Public", "b": "Easter", "c": "Christmas"}
)
fig, ax = plt.subplots()
ax.bar(range(len(holiday_sales)), holiday_sales.values, color="steelblue", edgecolor="none")
ax.set_xticks(range(len(holiday_sales)))
ax.set_xticklabels(holiday_sales.index)
ax.set(title="Mean sales by StateHoliday type", ylabel="Mean sales (€)")
plt.tight_layout()
plt.savefig(OUTPUTS / "08_stateholiday.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Promo analysis

# %% Promo frequency overall
promo_rate = open_df["Promo"].mean()
print(f"Promo=1 on {promo_rate:.1%} of open days")

# %% Promo frequency by month
promo_by_month = open_df.groupby("Month")["Promo"].mean()

fig, ax = plt.subplots()
ax.bar(promo_by_month.index, promo_by_month.values * 100, color="steelblue", edgecolor="none")
ax.set(title="Promo frequency by month (% of open days)",
       xlabel="Month", ylabel="% days with Promo=1")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
plt.tight_layout()
plt.savefig(OUTPUTS / "09_promo_by_month.png", bbox_inches="tight")
plt.show()

# %% Promo frequency by day of week
promo_by_dow = open_df.groupby("DayOfWeek")["Promo"].mean()

fig, ax = plt.subplots()
ax.bar(dow_labels, promo_by_dow.values * 100, color="steelblue", edgecolor="none")
ax.set(title="Promo frequency by day of week (% of open days)",
       ylabel="% days with Promo=1")
plt.tight_layout()
plt.savefig(OUTPUTS / "10_promo_by_dow.png", bbox_inches="tight")
plt.show()

# %% Promo frequency by StoreType
promo_by_type = open_df.groupby("StoreType")["Promo"].mean()

fig, ax = plt.subplots()
ax.bar(promo_by_type.index, promo_by_type.values * 100, color="steelblue", edgecolor="none")
ax.set(title="Promo frequency by StoreType (% of open days)",
       xlabel="StoreType", ylabel="% days with Promo=1")
plt.tight_layout()
plt.savefig(OUTPUTS / "11_promo_by_storetype.png", bbox_inches="tight")
plt.show()

# %% Naive promo effect — mean sales promo vs non-promo
promo_sales = open_df.groupby("Promo")["Sales"].mean()
naive_lift = promo_sales[1] - promo_sales[0]
naive_lift_pct = naive_lift / promo_sales[0] * 100
print(f"Mean sales (Promo=0): €{promo_sales[0]:,.0f}")
print(f"Mean sales (Promo=1): €{promo_sales[1]:,.0f}")
print(f"Naive lift: +€{naive_lift:,.0f} (+{naive_lift_pct:.1f}%)")
# TO REVIEW: this is the unadjusted estimate. Promotions are not randomly
# assigned — stores run promos on days when sales are expected to be high.
# Phase 3 uses propensity score methods to estimate the causal effect.

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["No promo", "Promo"], promo_sales.values, color=["#aec6cf", "steelblue"], edgecolor="none")
ax.set(title=f"Naive promo effect: +{naive_lift_pct:.1f}% (unadjusted)",
       ylabel="Mean daily sales (€)")
for i, v in enumerate(promo_sales.values):
    ax.text(i, v + 100, f"€{v:,.0f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUTS / "12_naive_promo_effect.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Correlations

# %% Correlation heatmap
num_cols = ["Sales", "Customers", "Promo", "SchoolHoliday", "CompetitionDistance"]
corr = open_df[num_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, annot_kws={"size": 10})
ax.set_title("Correlation matrix (open days)")
plt.tight_layout()
plt.savefig(OUTPUTS / "13_correlations.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
# Key findings for the modelling phase:
# - Strong weekly seasonality (Friday peak, Sunday low for most stores).
# - December spike and a smaller summer trough visible in monthly/weekly plots.
# - StoreType b has markedly higher mean sales than types a, c, d.
# - Promo runs on ~38% of open days; frequency is uniform across months but
#   nearly absent on Sundays — suggesting systematic assignment, not random.
# - Naive promo lift is substantial; Phase 3 will test whether this survives
#   propensity score adjustment.

print("\nAll outputs saved to outputs/")
