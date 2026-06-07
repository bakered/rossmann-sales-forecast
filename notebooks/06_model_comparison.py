# %% [markdown]
# # Model Comparison Charts
#
# Produces four publication-quality figures:
#   1. RMSPE league table — horizontal bar chart, all models, both scopes
#   2. Actual vs predicted — store 70, best model (LightGBM), full val period
#   3. Error distribution — % error violin by model family
#   4. SHAP feature importance — LightGBM global model (top 20 features)
#
# The script re-fits LightGBM (≈2 min) so charts are reproducible without
# loading saved model files. All RMSPE scores for slower models (OLS, RF, etc.)
# are pulled from the benchmark run in 04_models.py.

# %% Imports
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import lightgbm as lgb
import shap

os.chdir(Path(__file__).resolve().parent.parent)
Path("outputs").mkdir(exist_ok=True)

# %% RMSPE
def rmspe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

# %% ── Benchmark scores from 04_models.py run ──────────────────────────────
# All-stores RMSPE (global models evaluated on the full val set)
SCORES_ALL = {
    "Seasonal naive":         0.1767,
    "OLS structural (mult)":  0.2095,
    "OLS structural (linear)": 0.2302,
    "OLS predictive (mult)":  0.1525,
    "OLS predictive (linear)": 0.1607,
    "Random Forest":          0.1455,
    "LightGBM":               0.1379,
    "CatBoost":               0.1375,
    "MLP":                    0.1835,
}

# Store-70 RMSPE — single-store models (trained on store 70 only)
SCORES_S70_SINGLE = {
    "OLS structural (mult)":  0.1270,
    "OLS structural (linear)": 0.1348,
    "OLS predictive (mult)":  0.1252,
    "OLS predictive (linear)": 0.1266,
    "Random Forest":          0.1116,
    "XGBoost":                0.1111,
}

# Store-70 RMSPE — global models filtered to store 70
SCORES_S70_GLOBAL = {
    "Seasonal naive":         0.1449,
    "OLS structural (mult)":  0.1235,
    "OLS structural (linear)": 0.1250,
    "OLS predictive (mult)":  0.1267,
    "OLS predictive (linear)": 0.1270,
    "Random Forest (global)": 0.1184,
    "LightGBM (global)":      0.1015,
    "CatBoost (global)":      0.1027,
    "MLP (global)":           0.1388,
}

# %% Load data
def load(name):
    p = Path(f"data/processed/{name}")
    if p.with_suffix(".parquet").exists():
        return pd.read_parquet(p.with_suffix(".parquet"))
    return pd.read_csv(p.with_suffix(".csv"), parse_dates=["Date"])

train_raw = load("train_features")
val_raw   = load("val_features")
train = train_raw[(train_raw["Open"] == 1) & (train_raw["Sales"] > 0)].copy()
val   = val_raw[(val_raw["Open"] == 1)     & (val_raw["Sales"] > 0)].copy()

print(f"Train: {len(train):,}  Val: {len(val):,}")

FOCAL_STORE = 70
XMAS_COLS = [f"is_dec_{d:02d}" for d in list(range(15, 25)) + list(range(26, 32))] + ["is_jan_02"]
LAG_COLS  = ["sales_lag_same_cond", "sales_roll4_same_cond", "sales_roll8_same_cond", "store_trend_56"]

for col in ["StoreType", "Assortment"]:
    cat = pd.Categorical(train[col])
    train[col + "_enc"] = cat.codes
    val[col   + "_enc"] = pd.Categorical(val[col], categories=cat.categories).codes

GLOBAL_TREE_FEATS = (
    ["DayOfWeek", "Month", "Year", "WeekOfYear", "DayOfMonth",
     "Promo", "Promo2_active", "SchoolHoliday",
     "CompetitionDistance", "months_since_competitor_opened", "no_competitor",
     "comp_opened_last_6m", "days_since_comp_opened",
     "is_pre_holiday", "is_bridge_day", "is_month_end",
     "StoreType_enc", "Assortment_enc"]
    + XMAS_COLS + LAG_COLS
)

def prep_global_tree(df):
    d = df.copy()
    d["log_sales"] = np.log(d["Sales"])
    keep = GLOBAL_TREE_FEATS + ["log_sales", "Store", "Date"]
    d = d[keep].dropna()
    return d[GLOBAL_TREE_FEATS], d["log_sales"], d["Store"].values, d["Date"].values

X_tr, y_tr, _,          _           = prep_global_tree(train)
X_vl, y_vl, vstores_vl, vdates_vl   = prep_global_tree(val)

# %% ── Fit LightGBM (global) ──────────────────────────────────────────────
print("Fitting LightGBM...")
t0 = time.time()
dtrain = lgb.Dataset(X_tr, label=y_tr)
dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)
params = {
    "objective":          "regression",
    "metric":             "rmse",
    "learning_rate":      0.05,
    "num_leaves":         63,
    "min_child_samples":  50,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "verbose":            -1,
    "seed":               42,
}
lgb_model = lgb.train(
    params, dtrain, num_boost_round=500,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)],
)
elapsed = time.time() - t0
print(f"LightGBM done in {elapsed:.0f}s")

pred_lgb = np.exp(lgb_model.predict(X_vl))
act_g    = np.exp(y_vl)

sc_all = rmspe(act_g, pred_lgb)
mask70 = vstores_vl == FOCAL_STORE
sc_s70 = rmspe(act_g[mask70], pred_lgb[mask70])
SCORES_ALL["LightGBM"]          = sc_all    # overwrite with fresh run
SCORES_S70_GLOBAL["LightGBM (global)"] = sc_s70
print(f"LightGBM | all: {sc_all:.4f}  store {FOCAL_STORE}: {sc_s70:.4f}")

# %% ── Chart 1: RMSPE league table ───────────────────────────────────────
#
# Two side-by-side panels: all-stores (left) and store-70 (right).
# Colour-coded by model family.

FAMILY_COLOUR = {
    "Seasonal naive":         "#aaaaaa",
    "OLS structural":         "#4e9af1",
    "OLS predictive":         "#1565c0",
    "Random Forest":          "#43a047",
    "XGBoost":                "#2e7d32",
    "LightGBM":               "#f57f17",
    "CatBoost":               "#e65100",
    "MLP":                    "#8e24aa",
    "Prophet":                "#c62828",
}

def family_colour(label):
    for k, c in FAMILY_COLOUR.items():
        if k.lower() in label.lower():
            return c
    return "#555555"

def barh_panel(ax, scores, title, baseline_key="Seasonal naive"):
    items  = sorted(scores.items(), key=lambda x: x[1])   # best (lowest) at top
    labels = [k for k, _ in items]
    vals   = [v for _, v in items]
    colors = [family_colour(k) for k in labels]
    baseline = scores.get(baseline_key, None)

    bars = ax.barh(labels, vals, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    if baseline is not None:
        ax.axvline(baseline, color="#aaaaaa", lw=1.2, linestyle="--", label="Naive baseline")

    for bar, val in zip(bars, vals):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8)

    ax.set(title=title, xlabel="RMSPE (lower = better)")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    if baseline is not None:
        ax.legend(fontsize=8)

fig, (ax_all, ax_s70) = plt.subplots(1, 2, figsize=(16, 6))

# Combine single + global for store-70 panel — take best per model name
s70_combined = {}
for k, v in {**SCORES_S70_SINGLE, **SCORES_S70_GLOBAL}.items():
    base = k.replace(" (global)", "").replace(" (single)", "")
    if base not in s70_combined or v < s70_combined[base]:
        s70_combined[base] = v

barh_panel(ax_all, SCORES_ALL,    "All stores — global models")
barh_panel(ax_s70, s70_combined,  f"Store {FOCAL_STORE} — best per model family")

plt.suptitle("RMSPE comparison across model families", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/rmspe_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/rmspe_comparison.png")

# %% ── Chart 2: Actual vs predicted — store 70 ───────────────────────────

s70_dates  = vdates_vl[mask70]
s70_actual = act_g.values[mask70]
s70_pred   = pred_lgb[mask70]
s70_pct_err = (s70_pred - s70_actual) / s70_actual * 100

order = np.argsort(s70_dates)
d = s70_dates[order];  a = s70_actual[order];  p = s70_pred[order];  e = s70_pct_err[order]

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})

axes[0].plot(d, a, lw=1.2, color="steelblue", label="Actual")
axes[0].plot(d, p, lw=1.2, color="coral",     label=f"LightGBM (RMSPE {sc_s70:.4f})",
             linestyle="--", alpha=0.85)
axes[0].set(title=f"Store {FOCAL_STORE} — actual vs LightGBM (validation: May–Jul 2015)",
            ylabel="Daily sales (€)")
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
axes[0].legend(fontsize=9)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

clr = np.where(e >= 0, "coral", "steelblue")
axes[1].bar(d, e, color=clr, width=1)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set(ylabel="% error", xlabel="Date")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"outputs/actual_vs_pred_store{FOCAL_STORE}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: outputs/actual_vs_pred_store{FOCAL_STORE}.png")

# %% ── Chart 3: SHAP feature importance ──────────────────────────────────
#
# Use a stratified sample (up to 3,000 rows) to keep SHAP runtime under a minute.

print("Computing SHAP values (sample of val set)...")
rng      = np.random.default_rng(42)
shap_idx = rng.choice(len(X_vl), size=min(3000, len(X_vl)), replace=False)
X_shap   = X_vl.iloc[shap_idx]

explainer   = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_shap)
shap_abs    = np.abs(shap_values).mean(axis=0)

feat_imp = pd.Series(shap_abs, index=GLOBAL_TREE_FEATS).sort_values(ascending=False).head(20)

# Group xmas dummies into a single "Christmas dummies" bar
xmas_imp = feat_imp[feat_imp.index.isin(XMAS_COLS)].sum()
feat_imp  = feat_imp[~feat_imp.index.isin(XMAS_COLS)]
feat_imp  = pd.concat([feat_imp, pd.Series({"Christmas dummies (pooled)": xmas_imp})]).sort_values(ascending=False)

SHAP_COLOUR = {
    "sales_lag_same_cond":          "#f57f17",
    "sales_roll4_same_cond":        "#ffa000",
    "sales_roll8_same_cond":        "#ffca28",
    "store_trend_56":               "#ffe082",
    "DayOfWeek":                    "#1565c0",
    "Month":                        "#1976d2",
    "WeekOfYear":                   "#42a5f5",
    "Christmas dummies (pooled)":   "#c62828",
    "Promo":                        "#2e7d32",
    "CompetitionDistance":          "#6a1b9a",
    "months_since_competitor_opened": "#8e24aa",
}

def shap_colour(feat):
    return SHAP_COLOUR.get(feat, "#607d8b")

fig, ax = plt.subplots(figsize=(10, 6))
colors = [shap_colour(f) for f in feat_imp.index]
ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors[::-1],
        height=0.65, edgecolor="white", linewidth=0.4)
ax.set(title="LightGBM — mean |SHAP| feature importance (global model, val sample)",
       xlabel="Mean |SHAP value| (impact on log sales prediction)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.savefig("outputs/shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/shap_importance.png")

# %% ── Chart 4: % error distribution by model family ─────────────────────
#
# Re-compute % errors for LightGBM (already have them).
# For other models, re-run OLS predictive mult and seasonal naive
# to show distribution shape contrast without re-running all models.

from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

def get_naive_preds(train_df, val_df):
    lookback = pd.merge(val_df[["Store", "Date"]],
                        train_df[["Store", "Date", "Sales"]].rename(
                            columns={"Date": "lookup_date", "Sales": "pred"}),
                        how="left",
                        left_on=["Store", "Date"],
                        right_on=["Store", "lookup_date"])
    # Use 364-day lookback
    val_copy = val_df.copy()
    val_copy["lookup_date"] = val_copy["Date"] - pd.Timedelta(days=364)
    merged = val_copy.merge(
        train_df[["Store", "Date", "Sales"]].rename(columns={"Date": "lookup_date", "Sales": "pred_naive"}),
        on=["Store", "lookup_date"], how="left"
    )
    mask = merged["pred_naive"].notna() & (merged["Sales"] > 0)
    return merged.loc[mask, "pred_naive"].values, merged.loc[mask, "Sales"].values

print("\nComputing error distributions...")
naive_pred, naive_act = get_naive_preds(train, val)
naive_errs = (naive_pred - naive_act) / naive_act * 100

# OLS predictive mult errors
CAL_ONLY  = ["Promo2_active", "is_pre_holiday", "is_bridge_day", "is_month_end"]

def build_predictive(df):
    dow   = pd.get_dummies(df["DayOfWeek"], prefix="dow",   drop_first=True).astype(float)
    month = pd.get_dummies(df["Month"],     prefix="month", drop_first=True).astype(float)
    year  = (df["Year"] - 2013).rename("year_trend").astype(float)
    cal   = df[CAL_ONLY + XMAS_COLS].astype(float)
    lags  = df[LAG_COLS].apply(np.log).rename(columns={c: f"log_{c}" for c in LAG_COLS})
    feats = pd.concat([dow, month, year, cal, lags], axis=1)
    feats.insert(0, "const", 1.0)
    return feats

tr_f = build_predictive(train)
vl_f = build_predictive(val)
y_tr_ols = np.log(train["Sales"])
y_vl_ols = np.log(val["Sales"])

tr_mask = tr_f.notna().all(axis=1) & np.isfinite(tr_f).all(axis=1)
vl_mask = vl_f.notna().all(axis=1) & np.isfinite(vl_f).all(axis=1)

X_tr_ols = tr_f[tr_mask].reset_index(drop=True)
y_tr_ols = y_tr_ols[tr_mask].reset_index(drop=True)
X_vl_ols = vl_f[vl_mask].reindex(columns=X_tr_ols.columns, fill_value=0).reset_index(drop=True)
y_vl_ols = y_vl_ols[vl_mask].reset_index(drop=True)

ols_model  = sm.OLS(y_tr_ols, X_tr_ols).fit()
ols_pred   = np.exp(ols_model.predict(X_vl_ols))
ols_act    = np.exp(y_vl_ols)
ols_errs   = (ols_pred - ols_act) / ols_act * 100

lgb_errs_all = (pred_lgb - act_g) / act_g * 100

# Clip to ±100% for visual clarity
clip = 100
naive_c = np.clip(naive_errs, -clip, clip)
ols_c   = np.clip(ols_errs,   -clip, clip)
lgb_c   = np.clip(lgb_errs_all, -clip, clip)

fig, ax = plt.subplots(figsize=(11, 5))
vp = ax.violinplot([naive_c, ols_c, lgb_c],
                   positions=[1, 2, 3],
                   showmedians=True, showextrema=False, widths=0.6)
for patch, colour in zip(vp["bodies"], ["#aaaaaa", "#1565c0", "#f57f17"]):
    patch.set_facecolor(colour)
    patch.set_alpha(0.7)
vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(1.5)

ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.set(title="% prediction error distribution — validation set (clipped ±100%)",
       ylabel="% error  (pred − actual) / actual",
       xticks=[1, 2, 3],
       xticklabels=["Seasonal naive", "OLS predictive (mult)", "LightGBM"])
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/error_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/error_distribution.png")

# %% Summary
print("\n" + "=" * 55)
print("All charts saved to outputs/")
print("  rmspe_comparison.png")
print(f"  actual_vs_pred_store{FOCAL_STORE}.png")
print("  shap_importance.png")
print("  error_distribution.png")
print("=" * 55)
