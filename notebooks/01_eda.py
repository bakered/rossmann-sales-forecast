# %% [markdown]
# # Rossmann EDA — Feature Exploration

# %% Imports
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure working directory is always the project root, regardless of where
# VS Code launches the kernel from.
os.chdir(Path(__file__).resolve().parent.parent)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 4)

# %% [markdown]
# ## Feature descriptions
#
# **train.csv**
#
# | Column | Description |
# |---|---|
# | `Store` | Unique store ID (1–1115) |
# | `DayOfWeek` | 1=Monday … 7=Sunday |
# | `Date` | Date of the observation |
# | `Sales` | Turnover for that day — the target variable |
# | `Customers` | Number of customers that day |
# | `Open` | 0=closed, 1=open |
# | `Promo` | Whether the store ran a promotion that day |
# | `StateHoliday` | 0=none, a=public holiday, b=Easter, c=Christmas |
# | `SchoolHoliday` | Whether public schools were closed in that state |
#
# **store.csv** (merged on `Store`)
#
# | Column | Description |
# |---|---|
# | `StoreType` | Store format: a, b, c, d — Rossmann's internal segmentation. Type b is rare (~17 stores) but has markedly higher sales |
# | `Assortment` | Product range: a=basic, b=extra, c=extended |
# | `CompetitionDistance` | Distance in metres to nearest competitor. ~0.3% NaN — likely stores with no known nearby competitor |
# | `CompetitionOpenSinceMonth/Year` | When the nearest competitor opened. ~32% NaN |
# | `Promo2` | Whether the store participates in a recurring promotional scheme (0/1) |
# | `Promo2SinceWeek/Year` | When the store joined Promo2. ~50% NaN — non-participating stores |
# | `PromoInterval` | Months in which Promo2 runs, e.g. "Jan,Apr,Jul,Oct". ~50% NaN for same reason |

# %% Load data
train = pd.read_csv("data/raw/train.csv", parse_dates=["Date"],
                    dtype={"StateHoliday": str})
store = pd.read_csv("data/raw/store.csv")
test  = pd.read_csv("data/raw/test.csv",  parse_dates=["Date"])

# Merge store metadata onto train
df = train.merge(store, on="Store", how="left")

print("train shape:", train.shape)
print("store shape:", store.shape)
print("test shape: ", test.shape)
print("\nDate range:", df["Date"].min(), "→", df["Date"].max())

# %% Basic info — dtypes and nulls
print(df.dtypes)
print("\nNull counts:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# %% Descriptive stats — numeric columns
df.describe().T

# %% Descriptive stats — categorical columns
cat_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

# %% Sales distribution
fig, axes = plt.subplots(1, 2)
axes[0].hist(df.loc[df["Open"] == 1, "Sales"], bins=60, color="steelblue", edgecolor="none")
axes[0].set(title="Sales (open days)", xlabel="Sales", ylabel="Count")

axes[1].hist(np.log1p(df.loc[df["Open"] == 1, "Sales"]), bins=60, color="steelblue", edgecolor="none")
axes[1].set(title="log(Sales + 1)", xlabel="log Sales", ylabel="Count")

plt.tight_layout()
plt.show()

# %% Customers distribution
fig, axes = plt.subplots(1, 2)
axes[0].hist(df.loc[df["Open"] == 1, "Customers"], bins=60, color="coral", edgecolor="none")
axes[0].set(title="Customers (open days)", xlabel="Customers", ylabel="Count")

axes[1].hist(np.log1p(df.loc[df["Open"] == 1, "Customers"]), bins=60, color="coral", edgecolor="none")
axes[1].set(title="log(Customers + 1)", xlabel="log Customers", ylabel="Count")

plt.tight_layout()
plt.show()

# %% Sales over time — aggregate daily
daily = df.groupby("Date")["Sales"].sum().reset_index()
plt.plot(daily["Date"], daily["Sales"], lw=0.8, color="steelblue")
plt.title("Total daily sales across all stores")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# %% Seasonality — average sales by day of week
dow = df[df["Open"] == 1].groupby("DayOfWeek")["Sales"].mean()
dow.plot(kind="bar", color="steelblue", edgecolor="none")
plt.title("Mean sales by day of week (1=Mon, 7=Sun)")
plt.xlabel("Day of week")
plt.ylabel("Mean sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% Seasonality — average sales by month
df["Month"] = df["Date"].dt.month
month = df[df["Open"] == 1].groupby("Month")["Sales"].mean()
month.plot(kind="bar", color="steelblue", edgecolor="none")
plt.title("Mean sales by month")
plt.xlabel("Month")
plt.ylabel("Mean sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% Store-level variation — distribution of mean sales per store
store_mean = df[df["Open"] == 1].groupby("Store")["Sales"].mean()
plt.hist(store_mean, bins=50, color="steelblue", edgecolor="none")
plt.title("Distribution of mean daily sales across 1,115 stores")
plt.xlabel("Mean daily sales")
plt.ylabel("Number of stores")
plt.tight_layout()
plt.show()

print(f"Lowest mean sales store:  {store_mean.idxmin()} ({store_mean.min():.0f})")
print(f"Highest mean sales store: {store_mean.idxmax()} ({store_mean.max():.0f})")

# %% Categorical features — StoreType and Assortment
fig, axes = plt.subplots(1, 2)
df["StoreType"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="none")
axes[0].set(title="Store count by StoreType", xlabel="StoreType", ylabel="Count")
axes[0].tick_params(axis="x", rotation=0)

df["Assortment"].value_counts().plot(kind="bar", ax=axes[1], color="coral", edgecolor="none")
axes[1].set(title="Store count by Assortment", xlabel="Assortment", ylabel="Count")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()

# %% Sales by StoreType
df[df["Open"] == 1].groupby("StoreType")["Sales"].mean().plot(
    kind="bar", color="steelblue", edgecolor="none"
)
plt.title("Mean sales by StoreType")
plt.xlabel("StoreType")
plt.ylabel("Mean daily sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% Promo — frequency and sales lift
print("Promo=1 on", f"{df['Promo'].mean():.1%}", "of all rows")
print("Promo=1 on", f"{df[df['Open']==1]['Promo'].mean():.1%}", "of open days")

promo_sales = df[df["Open"] == 1].groupby("Promo")["Sales"].mean()
promo_sales.plot(kind="bar", color="steelblue", edgecolor="none")
plt.title("Mean sales: Promo=0 vs Promo=1")
plt.xlabel("Promo")
plt.ylabel("Mean sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% CompetitionDistance — distribution and nulls
print("CompetitionDistance nulls:", df["CompetitionDistance"].isnull().sum())
plt.hist(df["CompetitionDistance"].dropna(), bins=60, color="steelblue", edgecolor="none")
plt.title("CompetitionDistance distribution")
plt.xlabel("Distance (m)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# %% Correlation matrix — numeric columns
num_cols = ["Sales", "Customers", "Promo", "Open", "SchoolHoliday", "CompetitionDistance"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()
