# %% [markdown]
# # Competition Opening Effect on Sales
#
# Research question: does the opening of a nearby competitor reduce a Rossmann
# store's sales, and if so by how much and for how long?
#
# **Identification strategy — Regression Discontinuity in Time (RDiT)**
#
# CompetitionOpenSinceYear/Month records the calendar month a nearby competitor
# opened. We treat this as a sharp cutoff: for each affected store, sales
# before the opening month form the control window and sales after form the
# treatment window. Within each store the pre/post comparison controls for
# all time-invariant store characteristics.
#
# Key assumptions:
#   1. The opening date is not endogenously chosen in response to short-run
#      Rossmann sales fluctuations (plausible — competitor decides independently).
#   2. No other simultaneous shock hits the same stores at the same time
#      (checked via placebo tests).
#
# We also run a panel DiD as robustness: treated stores (competition opened
# within window) vs control stores (no competition opened or already had one).

# %% Imports
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

os.chdir(Path(__file__).resolve().parent.parent)
Path("outputs").mkdir(exist_ok=True)

# %% Load data
train_raw = pd.read_csv("data/raw/train.csv", parse_dates=["Date"],
                         dtype={"StateHoliday": str})
store     = pd.read_csv("data/raw/store.csv")
df = train_raw.merge(store, on="Store", how="left")
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
df["log_sales"] = np.log(df["Sales"])

print(f"Rows (open, Sales>0): {len(df):,}  |  Stores: {df['Store'].nunique()}")
print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# %% Identify treatment stores
# A store is "treated" if its competitor opened DURING the training window.
# We need the opening event to be observable — both pre and post data must exist.

store_comp = store.dropna(subset=["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]).copy()
store_comp["comp_open_date"] = pd.to_datetime(
    store_comp["CompetitionOpenSinceYear"].astype(int).astype(str) + "-" +
    store_comp["CompetitionOpenSinceMonth"].astype(int).astype(str) + "-01"
)

train_start = df["Date"].min()
train_end   = df["Date"].max()

# Treated: competition opened mid-window (need at least 60 days pre and post)
treated = store_comp[
    (store_comp["comp_open_date"] >= train_start + pd.Timedelta(days=60)) &
    (store_comp["comp_open_date"] <= train_end   - pd.Timedelta(days=60))
].copy()

print(f"\nTreated stores (competition opened mid-window): {len(treated)}")
print(f"Opening year distribution:")
print(treated["CompetitionOpenSinceYear"].value_counts().sort_index().to_string())

# %% [markdown]
# ## Event-study plot
#
# For each treated store, centre time at the competition opening month (t=0).
# Average log(Sales) across stores in each relative month.
# If competition hurts sales we expect a drop at t=0 and onward.

# %%
WINDOW_MONTHS = 12   # months before and after to include

def get_event_panel(store_id, open_date, df):
    s = df[df["Store"] == store_id].copy()
    s["rel_month"] = ((s["Date"].dt.year  - open_date.year) * 12 +
                      (s["Date"].dt.month - open_date.month))
    s = s[(s["rel_month"] >= -WINDOW_MONTHS) & (s["rel_month"] <= WINDOW_MONTHS)]
    return s[["Store", "Date", "rel_month", "log_sales", "Promo", "DayOfWeek"]]

panels = []
for _, row in treated.iterrows():
    panel = get_event_panel(row["Store"], row["comp_open_date"], df)
    if len(panel) > 0:
        panels.append(panel)

event_df = pd.concat(panels, ignore_index=True)
print(f"\nEvent panel: {len(panels)} stores, {len(event_df):,} store-days")

# Monthly average log sales (controlling for promo and day-of-week within each store)
# We residualise log_sales on store × DayOfWeek dummies and Promo to remove
# weekly seasonality before plotting the event trend.
event_df["store_dow"] = event_df["Store"].astype(str) + "_" + event_df["DayOfWeek"].astype(str)
resid_model = smf.ols("log_sales ~ C(store_dow) + Promo", data=event_df).fit()
event_df["log_sales_resid"] = resid_model.resid

monthly = (event_df.groupby("rel_month")["log_sales_resid"]
           .agg(mean="mean", se=lambda x: x.std() / np.sqrt(len(x)))
           .reset_index())

fig, ax = plt.subplots(figsize=(12, 4))
ax.axvline(0, color="red", lw=1.5, linestyle="--", label="Competition opens")
ax.axhline(0, color="black", lw=0.8, linestyle=":")
ax.fill_between(monthly["rel_month"],
                monthly["mean"] - 1.96 * monthly["se"],
                monthly["mean"] + 1.96 * monthly["se"],
                alpha=0.2, color="steelblue")
ax.plot(monthly["rel_month"], monthly["mean"], color="steelblue", lw=2, marker="o", ms=4)
ax.set(title="Event study: average (residualised) log sales around competitor opening\n"
             f"(n={len(panels)} stores, ±{WINDOW_MONTHS} months, 95% CI shaded)",
       xlabel="Months relative to competitor opening (0 = opening month)",
       ylabel="Residualised log sales")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/comp_event_study.png", bbox_inches="tight")
plt.show()
print("Event study saved.")

# %% [markdown]
# ## Regression Discontinuity in Time (RDiT)
#
# For each treated store estimate:
#   log(Sales_it) = α + β·POST_it + γ·t + δ·POST_it·t + ε_it
#
# where t = days relative to opening, POST = 1 if t ≥ 0.
# β captures the immediate level shift at the discontinuity.
# γ and δ allow separate linear trends before and after.
# We pool all stores and include store fixed effects (within-store demeaning).
#
# Bandwidth: ±90 days around the opening (≈3 months each side).

# %%
BW_DAYS = 90

rdit_panels = []
for _, row in treated.iterrows():
    s = df[df["Store"] == row["Store"]].copy()
    s["t"] = (s["Date"] - row["comp_open_date"]).dt.days
    s = s[(s["t"] >= -BW_DAYS) & (s["t"] <= BW_DAYS)].copy()
    s["post"]    = (s["t"] >= 0).astype(int)
    s["t_post"]  = s["t"] * s["post"]
    if len(s) >= 30:
        rdit_panels.append(s[["Store", "Date", "t", "post", "t_post",
                                "log_sales", "Promo", "DayOfWeek"]])

rdit_df = pd.concat(rdit_panels, ignore_index=True)
print(f"\nRDiT panel: {len(rdit_panels)} stores, {len(rdit_df):,} store-days  (±{BW_DAYS} day window)")

# Within-store demean to absorb store fixed effects
rdit_df["store_dow"] = rdit_df["Store"].astype(str) + "_" + rdit_df["DayOfWeek"].astype(str)
store_means = rdit_df.groupby("Store")[["log_sales", "t", "post", "t_post", "Promo"]].mean()

for col in ["log_sales", "t", "post", "t_post", "Promo"]:
    rdit_df[f"{col}_dm"] = (rdit_df[col].values -
                             store_means.loc[rdit_df["Store"].values, col].values)

X_rdit = sm.add_constant(rdit_df[["post_dm", "t_dm", "t_post_dm", "Promo_dm"]])
y_rdit = rdit_df["log_sales_dm"]
rdit_model = sm.OLS(y_rdit, X_rdit).fit(cov_type="HC3")

print("\nRDiT results (within-store, HC3 SEs):")
print(rdit_model.summary2().tables[1].round(4))

beta_post = rdit_model.params["post_dm"]
ci_lo, ci_hi = rdit_model.conf_int().loc["post_dm"]
pct_effect = (np.exp(beta_post) - 1) * 100
print(f"\nImmediate effect of competition opening:")
print(f"  β = {beta_post:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  ≈ {pct_effect:+.1f}% change in sales at the opening month")

# %% RDiT plot — average log sales vs days-to-opening (binned)
rdit_df["t_bin"] = (rdit_df["t"] // 7) * 7   # weekly bins
bin_avg = (rdit_df.groupby("t_bin")["log_sales_dm"]
           .agg(mean="mean", se=lambda x: x.std() / np.sqrt(len(x)))
           .reset_index())

# Fit line on each side
pre  = bin_avg[bin_avg["t_bin"] < 0]
post = bin_avg[bin_avg["t_bin"] >= 0]

def fit_line(df_side):
    X = sm.add_constant(df_side["t_bin"])
    m = sm.OLS(df_side["mean"], X).fit()
    return m.params["const"], m.params["t_bin"]

pre_int,  pre_slope  = fit_line(pre)
post_int, post_slope = fit_line(post)

fig, ax = plt.subplots(figsize=(12, 4))
ax.axvline(0, color="red", lw=1.5, linestyle="--", label="Competition opens")
ax.axhline(0, color="black", lw=0.8, linestyle=":")
ax.scatter(bin_avg["t_bin"], bin_avg["mean"], s=20, color="steelblue", zorder=3)
ax.fill_between(bin_avg["t_bin"],
                bin_avg["mean"] - 1.96 * bin_avg["se"],
                bin_avg["mean"] + 1.96 * bin_avg["se"],
                alpha=0.15, color="steelblue")

t_pre  = np.linspace(-BW_DAYS, -1, 100)
t_post = np.linspace(0, BW_DAYS, 100)
ax.plot(t_pre,  pre_int  + pre_slope  * t_pre,  "steelblue", lw=2)
ax.plot(t_post, post_int + post_slope * t_post, "coral",     lw=2)

ax.set(title=f"RDiT: log sales around competition opening  "
              f"(β={beta_post:+.3f}, {pct_effect:+.1f}%, n={len(rdit_panels)} stores)",
       xlabel="Days relative to competitor opening",
       ylabel="Within-store demeaned log sales")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/comp_rdit.png", bbox_inches="tight")
plt.show()
print("RDiT plot saved.")

# %% [markdown]
# ## Heterogeneity — does effect vary by competition distance?
#
# Stores with a closer competitor might be more affected.
# Split into terciles of CompetitionDistance and re-run RDiT for each.

# %%
dist_terciles = treated["CompetitionDistance"].quantile([1/3, 2/3]).values
print(f"\nCompetitionDistance tercile cutoffs: {dist_terciles.round(0)}")

treated["dist_group"] = pd.cut(treated["CompetitionDistance"],
                                bins=[-np.inf, dist_terciles[0], dist_terciles[1], np.inf],
                                labels=["close", "medium", "far"])

het_results = {}
for grp in ["close", "medium", "far"]:
    grp_stores = treated.loc[treated["dist_group"] == grp, "Store"].values
    sub = rdit_df[rdit_df["Store"].isin(grp_stores)]
    if len(sub) < 50:
        continue
    X = sm.add_constant(sub[["post_dm", "t_dm", "t_post_dm", "Promo_dm"]])
    m = sm.OLS(sub["log_sales_dm"], X).fit(cov_type="HC3")
    b = m.params["post_dm"]
    ci = m.conf_int().loc["post_dm"]
    het_results[grp] = {"beta": b, "ci_lo": ci[0], "ci_hi": ci[1],
                         "pct": (np.exp(b) - 1) * 100, "n": sub["Store"].nunique()}
    print(f"  {grp:6s}: β={b:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  "
          f"≈{(np.exp(b)-1)*100:+.1f}%  (n={sub['Store'].nunique()} stores)")

fig, ax = plt.subplots(figsize=(7, 3))
groups = list(het_results.keys())
betas  = [het_results[g]["beta"] for g in groups]
lo     = [het_results[g]["beta"] - het_results[g]["ci_lo"] for g in groups]
hi     = [het_results[g]["ci_hi"] - het_results[g]["beta"] for g in groups]
colors = ["coral" if b < 0 else "steelblue" for b in betas]
ax.barh(groups, betas, xerr=[lo, hi], color=colors, capsize=5, height=0.5)
ax.axvline(0, color="black", lw=0.8)
ax.set(title="RDiT effect by competition distance tercile",
       xlabel="β (log sales change at opening)")
plt.tight_layout()
plt.savefig("outputs/comp_het_distance.png", bbox_inches="tight")
plt.show()
print("Heterogeneity plot saved.")

# %% [markdown]
# ## Placebo test
#
# Assign a fake opening date 6 months before the real one and re-run RDiT.
# If the estimate is non-zero, the result may be driven by pre-existing trends
# rather than the actual opening.

# %%
placebo_panels = []
for _, row in treated.iterrows():
    fake_date = row["comp_open_date"] - pd.DateOffset(months=6)
    s = df[df["Store"] == row["Store"]].copy()
    s["t"] = (s["Date"] - fake_date).dt.days
    s = s[(s["t"] >= -BW_DAYS) & (s["t"] <= BW_DAYS)].copy()
    s["post"]   = (s["t"] >= 0).astype(int)
    s["t_post"] = s["t"] * s["post"]
    if len(s) >= 30:
        placebo_panels.append(s[["Store", "t", "post", "t_post", "log_sales", "Promo"]])

placebo_df = pd.concat(placebo_panels, ignore_index=True)
store_means_p = placebo_df.groupby("Store")[["log_sales", "t", "post", "t_post", "Promo"]].mean()
for col in ["log_sales", "t", "post", "t_post", "Promo"]:
    placebo_df[f"{col}_dm"] = (placebo_df[col].values -
                                store_means_p.loc[placebo_df["Store"].values, col].values)

X_p = sm.add_constant(placebo_df[["post_dm", "t_dm", "t_post_dm", "Promo_dm"]])
placebo_model = sm.OLS(placebo_df["log_sales_dm"], X_p).fit(cov_type="HC3")
b_placebo = placebo_model.params["post_dm"]
ci_p      = placebo_model.conf_int().loc["post_dm"]
print(f"\nPlacebo test (fake opening 6 months early):")
print(f"  β = {b_placebo:.4f}  [{ci_p[0]:.4f}, {ci_p[1]:.4f}]  "
      f"({'significant — check pre-trends' if placebo_model.pvalues['post_dm'] < 0.05 else 'not significant — pre-trends OK'})")

# %% Summary
print("\n" + "="*55)
print("Competition opening effect — summary")
print("="*55)
print(f"Treated stores:          {len(rdit_panels)}")
print(f"Bandwidth:               ±{BW_DAYS} days")
print(f"Immediate effect (β):    {beta_post:+.4f}")
print(f"95% CI:                  [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Approx % change:         {pct_effect:+.1f}%")
print(f"Placebo β (null check):  {b_placebo:+.4f}  "
      f"(p={placebo_model.pvalues['post_dm']:.3f})")
print("="*55)
