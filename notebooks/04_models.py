# %% [markdown]
# # Models
#
# Comparison matrix
# | | Structural (calendar + promo) | Predictive (lag features) |
# |---|---|---|
# | Multiplicative (log target) | OLS-SM | OLS-PM |
# | Linear (raw target) | OLS-SL | OLS-PL |
# | Tree-based | — | RF / XGBoost |
#
# Structural: store FE absorb baseline; coefficients are clean multipliers / additive €.
# Predictive: lag features absorb store baseline and recent trend; no Promo (encoded in lag).
# Multiplicative: log(Sales) target — % errors scale with store size.
# Linear: raw Sales target — absolute errors dominate; expected to underperform on RMSPE.
#
# Evaluated on two scopes:
#   (a) Store 70 — single-store models trained on store 70 only; global models filtered.
#   (b) All stores — global models only.

# %% Imports
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

os.chdir(Path(__file__).resolve().parent.parent)
Path("outputs").mkdir(exist_ok=True)

# %% RMSPE
def rmspe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

assert abs(rmspe([100, 200], [110, 180]) - 0.1) < 0.001
print("RMSPE OK")

# %% Load data
def load(name):
    p = Path(f"data/processed/{name}")
    if p.with_suffix(".parquet").exists():
        return pd.read_parquet(p.with_suffix(".parquet"))
    return pd.read_csv(p.with_suffix(".csv"), parse_dates=["Date"])

train = load("train_features")
val   = load("val_features")
train = train[(train["Open"] == 1) & (train["Sales"] > 0)].copy()
val   = val[(val["Open"] == 1)   & (val["Sales"] > 0)].copy()

print(f"Train: {len(train):,}  |  {train['Date'].min().date()} → {train['Date'].max().date()}")
print(f"Val:   {len(val):,}   |  {val['Date'].min().date()} → {val['Date'].max().date()}")

store_means  = train.groupby("Store")["Sales"].mean().sort_values()
FOCAL_STORE  = store_means.index[len(store_means) // 2]
print(f"Focal store: {FOCAL_STORE}  (mean sales €{store_means[FOCAL_STORE]:,.0f})")

# score trackers
scores_s70  = {}   # store 70 results
scores_all  = {}   # all-stores results

# %% Feature column lists
XMAS_COLS = [f"is_dec_{d:02d}" for d in list(range(15, 25)) + list(range(26, 32))] + ["is_jan_02"]
CAL_PROMO = ["Promo", "Promo2_active", "is_pre_holiday", "is_bridge_day", "is_month_end"]
CAL_ONLY  = ["Promo2_active", "is_pre_holiday", "is_bridge_day", "is_month_end"]
LAG_COLS  = ["sales_lag_same_cond", "sales_roll4_same_cond", "sales_roll8_same_cond", "store_trend_56"]

# %% [markdown]
# ## Seasonal naive baseline
#
# For each val row predict sales from 52 weeks (364 days) prior — same store,
# same day-of-week, same Promo status. Requires no model fitting.

# %%
train_lookup = (
    train[["Store", "Date", "DayOfWeek", "Promo", "Sales"]]
    .rename(columns={"Date": "lookup_date", "Sales": "pred_naive"})
)
val_naive = val.copy()
val_naive["lookup_date"] = val_naive["Date"] - pd.Timedelta(days=364)
val_naive = val_naive.merge(train_lookup, on=["Store", "lookup_date", "DayOfWeek", "Promo"], how="left")

# Fallback for unmatched rows: store-level training mean
store_mean_fallback = train.groupby("Store")["Sales"].mean()
val_naive["pred_naive"] = val_naive.apply(
    lambda r: r["pred_naive"] if pd.notna(r["pred_naive"])
              else store_mean_fallback.get(r["Store"], np.nan),
    axis=1,
)

sc_naive_all = rmspe(val_naive["Sales"], val_naive["pred_naive"])
sc_naive_s70 = rmspe(
    val_naive.loc[val_naive["Store"] == FOCAL_STORE, "Sales"],
    val_naive.loc[val_naive["Store"] == FOCAL_STORE, "pred_naive"],
)
scores_all["Seasonal naive"]  = sc_naive_all
scores_s70["Seasonal naive"]  = sc_naive_s70
print(f"Seasonal naive — all stores: {sc_naive_all:.4f}  |  store {FOCAL_STORE}: {sc_naive_s70:.4f}")

# %% [markdown]
# ## Single-store OLS models — store FOCAL_STORE
#
# Four OLS variants: {structural, predictive} × {multiplicative, linear}
# No store fixed effects — one store = one intercept = the constant.
# Structural: DayOfWeek, Month, Year, Promo, Promo2_active, calendar flags, Xmas dummies.
# Predictive: log-lags (mult) or raw lags (linear) replace Promo.

# %%
def build_store_ols(df, store_id, use_lags=False, log_target=True):
    s = df[df["Store"] == store_id].reset_index(drop=True).copy()
    dow   = pd.get_dummies(s["DayOfWeek"], prefix="dow",   drop_first=True).astype(float)
    month = pd.get_dummies(s["Month"],     prefix="month", drop_first=True).astype(float)
    year  = (s["Year"] - 2013).rename("year_trend").astype(float)

    if use_lags:
        if log_target:
            lag_feats = s[LAG_COLS].apply(np.log).rename(columns={c: f"log_{c}" for c in LAG_COLS})
        else:
            lag_feats = s[LAG_COLS].copy()
        cal_feats = s[CAL_ONLY + XMAS_COLS].astype(float)
        feats = pd.concat([dow, month, year, cal_feats, lag_feats], axis=1)
    else:
        feats = pd.concat([dow, month, year, s[CAL_PROMO + XMAS_COLS].astype(float)], axis=1)

    feats.insert(0, "const", 1.0)
    y = np.log(s["Sales"]) if log_target else s["Sales"].astype(float)
    mask = feats.notna().all(axis=1) & np.isfinite(feats).all(axis=1) & y.notna()
    return feats[mask], y[mask]

for use_lags, log_tgt, label in [
    (False, True,  "OLS structural mult"),
    (False, False, "OLS structural linear"),
    (True,  True,  "OLS predictive mult"),
    (True,  False, "OLS predictive linear"),
]:
    X_tr, y_tr = build_store_ols(train, FOCAL_STORE, use_lags=use_lags, log_target=log_tgt)
    X_vl, y_vl = build_store_ols(val,   FOCAL_STORE, use_lags=use_lags, log_target=log_tgt)
    X_vl = X_vl.reindex(columns=X_tr.columns, fill_value=0)

    zv = X_tr.columns[(X_tr.std() == 0) & (X_tr.columns != "const")]
    X_tr, X_vl = X_tr.drop(columns=zv), X_vl.drop(columns=zv)

    model = sm.OLS(y_tr, X_tr).fit()
    pred  = model.predict(X_vl)

    if log_tgt:
        sc = rmspe(np.exp(y_vl), np.exp(pred))
    else:
        sc = rmspe(y_vl, pred)

    scores_s70[label] = sc
    print(f"{label} | store {FOCAL_STORE}: {sc:.4f}  (R²={model.rsquared:.3f})")

    # Print structural multiplicative coefficients for interpretability
    if not use_lags and log_tgt:
        coef_df = pd.DataFrame({
            "coef":       model.params,
            "multiplier": np.exp(model.params),
            "p_value":    model.pvalues,
        }).round(4)
        print(f"\nStore {FOCAL_STORE} structural OLS — multiplicative coefficients:")
        print(coef_df.to_string())

# %% [markdown]
# ## Single-store tree models — store FOCAL_STORE
#
# RF and XGBoost trained on store FOCAL_STORE only.
# Use full feature set (calendar + promo + lags); target = log(Sales).
# Trees handle raw lag scales without log-transformation.

# %%
TREE_FEATS = (
    ["DayOfWeek", "Month", "Year", "WeekOfYear", "DayOfMonth",
     "Promo", "Promo2_active", "SchoolHoliday",
     "is_pre_holiday", "is_bridge_day", "is_month_end"]
    + XMAS_COLS + LAG_COLS
)

def prep_store_tree(df, store_id):
    d = df[df["Store"] == store_id].copy()
    d["log_sales"] = np.log(d["Sales"])
    keep = TREE_FEATS + ["log_sales"]
    d = d[keep].dropna()
    return d[TREE_FEATS], d["log_sales"]

X_tr_t, y_tr_t = prep_store_tree(train, FOCAL_STORE)
X_vl_t, y_vl_t = prep_store_tree(val,   FOCAL_STORE)

# RF single-store
rf_s70 = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
rf_s70.fit(X_tr_t, y_tr_t)
pred_rf_s70 = np.exp(rf_s70.predict(X_vl_t))
sc = rmspe(np.exp(y_vl_t), pred_rf_s70)
scores_s70["Random Forest"] = sc
print(f"Random Forest | store {FOCAL_STORE}: {sc:.4f}")

# XGBoost single-store
dtrain_s70 = xgb.DMatrix(X_tr_t, label=y_tr_t)
dval_s70   = xgb.DMatrix(X_vl_t, label=y_vl_t)
xgb_params = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "learning_rate": 0.05, "max_depth": 6,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 3, "seed": 42,
}
xgb_s70 = xgb.train(xgb_params, dtrain_s70, num_boost_round=500,
                     evals=[(dval_s70, "val")], early_stopping_rounds=30,
                     verbose_eval=False)
pred_xgb_s70 = np.exp(xgb_s70.predict(dval_s70))
sc = rmspe(np.exp(y_vl_t), pred_xgb_s70)
scores_s70["XGBoost"] = sc
print(f"XGBoost       | store {FOCAL_STORE}: {sc:.4f}")

# %% [markdown]
# ## Global OLS models — all stores
#
# Structural: within-store demeaning (Frisch-Waugh) ≡ store fixed effects,
#             avoids constructing the 1,114-column dummy matrix.
# Predictive: log-lags (mult) or raw lags (linear) replace store dummies and Promo.

# %%
def _demean(df, feature_cols, log_target, store_col_means=None, store_y_means=None):
    d     = df.reset_index(drop=True)
    feats = feature_cols.reset_index(drop=True)   # align to same 0..N-1 index
    y     = np.log(d["Sales"]) if log_target else d["Sales"].astype(float)
    stores = d["Store"]

    mask  = feats.notna().all(axis=1) & np.isfinite(feats).all(axis=1) & y.notna()
    feats, y, stores = feats[mask].copy(), y[mask].copy(), stores[mask]

    if store_col_means is None:
        tmp = feats.copy()
        tmp["__y__"]     = y.values
        tmp["__store__"] = stores.values
        grp             = tmp.groupby("__store__")[list(tmp.columns)].mean()
        store_y_means   = grp["__y__"]
        store_col_means = grp.drop(columns=["__y__", "__store__"])
        feats = feats.drop(columns=[], errors="ignore")
    else:
        feats = feats.reindex(columns=store_col_means.columns, fill_value=0.0)

    si          = stores.values
    col_arr     = store_col_means.reindex(si).values
    y_arr       = store_y_means.reindex(si).values

    feats_dm = pd.DataFrame(feats.values - col_arr, columns=store_col_means.columns)
    y_dm     = pd.Series(y.values - y_arr)
    return feats_dm, y_dm, store_col_means, store_y_means, si, y_arr

def build_structural_features(df):
    dow   = pd.get_dummies(df["DayOfWeek"], prefix="dow",   drop_first=True).astype(float)
    month = pd.get_dummies(df["Month"],     prefix="month", drop_first=True).astype(float)
    year  = (df["Year"] - 2013).rename("year_trend").astype(float)
    return pd.concat([dow, month, year, df[CAL_PROMO + XMAS_COLS].astype(float)], axis=1)

def global_structural(train_df, val_df, log_target, label):
    tr_feats = build_structural_features(train_df)
    vl_feats = build_structural_features(val_df)

    X_tr, y_tr, scm, sym, _, _ = _demean(train_df, tr_feats, log_target)
    X_vl, y_vl, _,   _,  vstores, vy_arr = _demean(val_df, vl_feats, log_target, scm, sym)

    model = sm.OLS(y_tr, X_tr).fit()

    pred_dm = model.predict(X_vl).values
    pred_raw = pred_dm + sym.reindex(vstores).values        # add back store mean
    act_raw  = y_vl.values + vy_arr

    if log_target:
        pred_sales, act_sales = np.exp(pred_raw), np.exp(act_raw)
    else:
        pred_sales, act_sales = pred_raw, act_raw

    sc_all = rmspe(act_sales, pred_sales)
    mask70 = vstores == FOCAL_STORE
    sc_s70 = rmspe(act_sales[mask70], pred_sales[mask70])
    scores_all[label] = sc_all
    scores_s70[label] = sc_s70
    print(f"{label} | all: {sc_all:.4f}  store {FOCAL_STORE}: {sc_s70:.4f}  (within R²={model.rsquared:.3f})")

    if log_target:
        non_zero_var = model.params.index[X_tr.std() > 0]
        coef_df = pd.DataFrame({
            "coef":       model.params[non_zero_var],
            "multiplier": np.exp(model.params[non_zero_var]),
            "p_value":    model.pvalues[non_zero_var],
        }).round(4)
        print(f"\n{label} coefficients:")
        print(coef_df.to_string())

print("=== Global structural OLS ===")
global_structural(train, val, log_target=True,  label="OLS structural mult")
global_structural(train, val, log_target=False, label="OLS structural linear")

# %% Global predictive OLS
def build_predictive_features(df, log_lags):
    dow   = pd.get_dummies(df["DayOfWeek"], prefix="dow",   drop_first=True).astype(float)
    month = pd.get_dummies(df["Month"],     prefix="month", drop_first=True).astype(float)
    year  = (df["Year"] - 2013).rename("year_trend").astype(float)
    cal   = df[CAL_ONLY + XMAS_COLS].astype(float)
    if log_lags:
        lags = df[LAG_COLS].apply(np.log).rename(columns={c: f"log_{c}" for c in LAG_COLS})
    else:
        lags = df[LAG_COLS].copy()
    feats = pd.concat([dow, month, year, cal, lags], axis=1)
    feats.insert(0, "const", 1.0)
    return feats

def global_predictive(train_df, val_df, log_target, log_lags, label):
    tr_feats = build_predictive_features(train_df, log_lags)
    vl_feats = build_predictive_features(val_df,   log_lags)

    y_tr = np.log(train_df["Sales"]) if log_target else train_df["Sales"].astype(float)
    y_vl = np.log(val_df["Sales"])   if log_target else val_df["Sales"].astype(float)

    tr_mask = tr_feats.notna().all(axis=1) & np.isfinite(tr_feats).all(axis=1) & y_tr.notna()
    vl_mask = vl_feats.notna().all(axis=1) & np.isfinite(vl_feats).all(axis=1) & y_vl.notna()

    X_tr, y_tr = tr_feats[tr_mask].reset_index(drop=True), y_tr[tr_mask].reset_index(drop=True)
    X_vl = vl_feats[vl_mask].reindex(columns=X_tr.columns, fill_value=0).reset_index(drop=True)
    y_vl = y_vl[vl_mask].reset_index(drop=True)
    vstores = val_df.loc[vl_mask, "Store"].values

    model = sm.OLS(y_tr, X_tr).fit()
    pred  = model.predict(X_vl)

    if log_target:
        pred_sales = np.exp(pred)
        act_sales  = np.exp(y_vl)
    else:
        pred_sales = pred
        act_sales  = y_vl

    sc_all = rmspe(act_sales, pred_sales)
    mask70 = vstores == FOCAL_STORE
    sc_s70 = rmspe(act_sales.values[mask70], pred_sales.values[mask70])
    scores_all[label] = sc_all
    scores_s70[label] = sc_s70
    print(f"{label} | all: {sc_all:.4f}  store {FOCAL_STORE}: {sc_s70:.4f}  (R²={model.rsquared:.3f})")

print("\n=== Global predictive OLS ===")
global_predictive(train, val, log_target=True,  log_lags=True,  label="OLS predictive mult")
global_predictive(train, val, log_target=False, log_lags=False, label="OLS predictive linear")

# %% [markdown]
# ## Global tree models — Random Forest and XGBoost
#
# Store-level effects captured through lag/rolling features and store metadata.
# No store ID as a feature — should generalise to stores not in training.
# Target: log(Sales) for both RF and XGBoost.

# %%
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
    keep = GLOBAL_TREE_FEATS + ["log_sales", "Store"]
    d = d[keep].dropna()
    return d[GLOBAL_TREE_FEATS], d["log_sales"], d["Store"].values

X_tr_g, y_tr_g, _           = prep_global_tree(train)
X_vl_g, y_vl_g, vstores_g  = prep_global_tree(val)

print(f"\nGlobal tree train: {len(X_tr_g):,} rows | val: {len(X_vl_g):,} rows")

# %% Random Forest global
print("Fitting global Random Forest...")
rf_global = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=50,
                                   random_state=42, n_jobs=-1)
rf_global.fit(X_tr_g, y_tr_g)
pred_rf_g = np.exp(rf_global.predict(X_vl_g))
act_g     = np.exp(y_vl_g)

sc_all = rmspe(act_g, pred_rf_g)
mask70 = vstores_g == FOCAL_STORE
sc_s70 = rmspe(act_g.values[mask70], pred_rf_g[mask70])
scores_all["Random Forest"] = sc_all
scores_s70["Random Forest (global)"] = sc_s70
print(f"Random Forest  | all: {sc_all:.4f}  store {FOCAL_STORE}: {sc_s70:.4f}")

# %% XGBoost global
print("Fitting global XGBoost...")
dtrain_g = xgb.DMatrix(X_tr_g, label=y_tr_g)
dval_g   = xgb.DMatrix(X_vl_g, label=y_vl_g)
xgb_params = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "learning_rate": 0.05, "max_depth": 6,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 50, "seed": 42,
}
xgb_global = xgb.train(xgb_params, dtrain_g, num_boost_round=500,
                        evals=[(dval_g, "val")], early_stopping_rounds=30,
                        verbose_eval=50)
pred_xgb_g = np.exp(xgb_global.predict(dval_g))

sc_all = rmspe(act_g, pred_xgb_g)
sc_s70 = rmspe(act_g.values[mask70], pred_xgb_g[mask70])
scores_all["XGBoost"] = sc_all
scores_s70["XGBoost (global)"] = sc_s70
print(f"XGBoost        | all: {sc_all:.4f}  store {FOCAL_STORE}: {sc_s70:.4f}")

fi = pd.Series(xgb_global.get_score(importance_type="gain")).sort_values(ascending=False)
print("\nTop 15 XGBoost features by gain:")
print(fi.head(15).round(1).to_string())

# %% [markdown]
# ## LightGBM
#
# Gradient boosting with histogram-based splits — typically faster than XGBoost
# and competitive on tabular data. Same feature set and log(Sales) target.

# %%
import lightgbm as lgb

def fit_lgb(X_tr, y_tr, X_vl, y_vl, min_child_samples=50, label="LightGBM"):
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)
    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "learning_rate":    0.05,
        "num_leaves":       63,
        "min_child_samples": min_child_samples,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "verbose":          -1,
        "seed":             42,
    }
    cb = lgb.train(params, dtrain, num_boost_round=500,
                   valid_sets=[dval],
                   callbacks=[lgb.early_stopping(30, verbose=False),
                               lgb.log_evaluation(50)])
    return cb

# Single-store
lgb_s70 = fit_lgb(X_tr_t, y_tr_t, X_vl_t, y_vl_t, min_child_samples=5)
pred_lgb_s70 = np.exp(lgb_s70.predict(X_vl_t))
sc = rmspe(np.exp(y_vl_t), pred_lgb_s70)
scores_s70["LightGBM"] = sc
print(f"LightGBM | store {FOCAL_STORE}: {sc:.4f}")

# Global
print("Fitting global LightGBM...")
lgb_global = fit_lgb(X_tr_g, y_tr_g, X_vl_g, y_vl_g)
pred_lgb_g = np.exp(lgb_global.predict(X_vl_g))

sc_all = rmspe(act_g, pred_lgb_g)
sc_s70 = rmspe(act_g.values[mask70], pred_lgb_g[mask70])
scores_all["LightGBM"] = sc_all
scores_s70["LightGBM (global)"] = sc_s70
print(f"LightGBM | all: {sc_all:.4f}  store {FOCAL_STORE}: {sc_s70:.4f}")

# %% [markdown]
# ## CatBoost
#
# Gradient boosting with native categorical support — passes StoreType and
# Assortment as raw strings, no label encoding needed.
# Symmetric trees reduce overfitting; built-in ordered boosting reduces target leakage.

# %%
from catboost import CatBoostRegressor, Pool

CAT_FEATS_RAW = (
    ["DayOfWeek", "Month", "Year", "WeekOfYear", "DayOfMonth",
     "Promo", "Promo2_active", "SchoolHoliday",
     "CompetitionDistance", "months_since_competitor_opened", "no_competitor",
     "is_pre_holiday", "is_bridge_day", "is_month_end",
     "StoreType", "Assortment"]          # raw strings — CatBoost handles them
    + XMAS_COLS + LAG_COLS
)
CAT_COL_IDX = [CAT_FEATS_RAW.index("StoreType"), CAT_FEATS_RAW.index("Assortment")]

TREE_FEATS_SINGLE = (                   # single-store: StoreType/Assortment are constant, drop them
    ["DayOfWeek", "Month", "Year", "WeekOfYear", "DayOfMonth",
     "Promo", "Promo2_active", "SchoolHoliday",
     "is_pre_holiday", "is_bridge_day", "is_month_end"]
    + XMAS_COLS + LAG_COLS
)

def prep_cat(df, feat_cols, cat_idx=None):
    d = df.copy()
    d["log_sales"] = np.log(d["Sales"])
    keep = feat_cols + ["log_sales", "Store"]
    d = d[keep].dropna()
    X = d[feat_cols]
    y = d["log_sales"]
    stores = d["Store"].values
    pool = Pool(X, label=y, cat_features=cat_idx or [])
    return pool, y, stores

# Single-store (no categorical store features needed — constant per store)
def prep_cat_single(df, store_id):
    d = df[df["Store"] == store_id].copy()
    d["log_sales"] = np.log(d["Sales"])
    keep = TREE_FEATS_SINGLE + ["log_sales"]
    d = d[keep].dropna()
    return Pool(d[TREE_FEATS_SINGLE], label=d["log_sales"]), d["log_sales"]

pool_tr_s70, y_cat_tr_s70 = prep_cat_single(train, FOCAL_STORE)
pool_vl_s70, y_cat_vl_s70 = prep_cat_single(val,   FOCAL_STORE)

cb_s70 = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                            loss_function="RMSE", random_seed=42, verbose=False)
cb_s70.fit(pool_tr_s70, eval_set=pool_vl_s70, early_stopping_rounds=30)
pred_cb_s70 = np.exp(cb_s70.predict(pool_vl_s70))
sc = rmspe(np.exp(y_cat_vl_s70), pred_cb_s70)
scores_s70["CatBoost"] = sc
print(f"CatBoost | store {FOCAL_STORE}: {sc:.4f}")

# Global
print("Fitting global CatBoost...")
pool_tr_g, y_cat_tr_g, _         = prep_cat(train, CAT_FEATS_RAW, CAT_COL_IDX)
pool_vl_g, y_cat_vl_g, vstores_c = prep_cat(val,   CAT_FEATS_RAW, CAT_COL_IDX)

cb_global = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                               loss_function="RMSE", random_seed=42, verbose=100)
cb_global.fit(pool_tr_g, eval_set=pool_vl_g, early_stopping_rounds=30)
pred_cb_g = np.exp(cb_global.predict(pool_vl_g))
act_cb_g  = np.exp(y_cat_vl_g)

sc_all = rmspe(act_cb_g, pred_cb_g)
mask70_c = vstores_c == FOCAL_STORE
sc_s70 = rmspe(act_cb_g.values[mask70_c], pred_cb_g[mask70_c])
scores_all["CatBoost"] = sc_all
scores_s70["CatBoost (global)"] = sc_s70
print(f"CatBoost | all: {sc_all:.4f}  store {FOCAL_STORE}: {sc_s70:.4f}")

# %% [markdown]
# ## MLP (neural network)
#
# Feedforward neural network via sklearn MLPRegressor.
# Features must be standardised — tree models don't need this but NNs do.
# Architecture: two hidden layers (256, 128). Target: log(Sales).

# %%
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def prep_mlp(df, store_id=None):
    d = df if store_id is None else df[df["Store"] == store_id].copy()
    feat_cols = TREE_FEATS_SINGLE if store_id else GLOBAL_TREE_FEATS
    d = d.copy()
    d["log_sales"] = np.log(d["Sales"])
    keep = feat_cols + ["log_sales", "Store"]
    d = d[keep].dropna()
    return d[feat_cols].values.astype(float), d["log_sales"].values, d["Store"].values

# Single-store
X_mlp_tr_s, y_mlp_tr_s, _ = prep_mlp(train, FOCAL_STORE)
X_mlp_vl_s, y_mlp_vl_s, _ = prep_mlp(val,   FOCAL_STORE)

scaler_s = StandardScaler().fit(X_mlp_tr_s)
mlp_s70 = MLPRegressor(hidden_layer_sizes=(256, 128), activation="relu",
                        max_iter=200, learning_rate_init=1e-3,
                        early_stopping=True, validation_fraction=0.1,
                        random_state=42, verbose=False)
mlp_s70.fit(scaler_s.transform(X_mlp_tr_s), y_mlp_tr_s)
pred_mlp_s70 = np.exp(mlp_s70.predict(scaler_s.transform(X_mlp_vl_s)))
sc = rmspe(np.exp(y_mlp_vl_s), pred_mlp_s70)
scores_s70["MLP"] = sc
print(f"MLP | store {FOCAL_STORE}: {sc:.4f}")

# Global
print("Fitting global MLP...")
X_mlp_tr_g, y_mlp_tr_g, _          = prep_mlp(train)
X_mlp_vl_g, y_mlp_vl_g, vstores_m  = prep_mlp(val)

scaler_g = StandardScaler().fit(X_mlp_tr_g)
mlp_global = MLPRegressor(hidden_layer_sizes=(256, 128), activation="relu",
                           max_iter=200, learning_rate_init=1e-3,
                           early_stopping=True, validation_fraction=0.1,
                           random_state=42, verbose=False)
mlp_global.fit(scaler_g.transform(X_mlp_tr_g), y_mlp_tr_g)
pred_mlp_g = np.exp(mlp_global.predict(scaler_g.transform(X_mlp_vl_g)))
act_mlp_g  = np.exp(y_mlp_vl_g)

sc_all = rmspe(act_mlp_g, pred_mlp_g)
mask70_m = vstores_m == FOCAL_STORE
sc_s70 = rmspe(act_mlp_g[mask70_m], pred_mlp_g[mask70_m])
scores_all["MLP"] = sc_all
scores_s70["MLP (global)"] = sc_s70
print(f"MLP | all: {sc_all:.4f}  store {FOCAL_STORE}: {sc_s70:.4f}")

# %% [markdown]
# ## Prophet
#
# Additive decomposition: trend + weekly seasonality + yearly seasonality + regressors.
# Fitted separately per store (Prophet is a single-series model).
# Regressors: Promo, SchoolHoliday.
# Target: log(Sales) — keeps multiplicative interpretation and reduces skew.
# All 1,115 stores are fitted in parallel via joblib.

# %%
from prophet import Prophet
from joblib import Parallel, delayed
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

def fit_prophet_store(store_id, train_df, val_df):
    tr = train_df[train_df["Store"] == store_id].copy()
    vl = val_df[val_df["Store"] == store_id].copy()
    if len(tr) < 30 or len(vl) == 0:
        return store_id, np.nan

    tr_p = tr[["Date", "Sales", "Promo", "SchoolHoliday"]].rename(columns={"Date": "ds"})
    tr_p["y"] = np.log(tr_p["Sales"])

    m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                daily_seasonality=False, seasonality_mode="additive")
    m.add_regressor("Promo")
    m.add_regressor("SchoolHoliday")
    m.fit(tr_p[["ds", "y", "Promo", "SchoolHoliday"]], algorithm="Newton")

    future = vl[["Date", "Promo", "SchoolHoliday"]].rename(columns={"Date": "ds"})
    forecast = m.predict(future)
    pred = np.exp(forecast["yhat"].values)
    actual = vl["Sales"].values
    sc = rmspe(actual, pred)
    return store_id, sc

print("Fitting Prophet (one model per store, parallel)...")
store_ids = train["Store"].unique()
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(fit_prophet_store)(sid, train, val) for sid in store_ids
)
prophet_scores = {sid: sc for sid, sc in results if not np.isnan(sc)}

sc_all = np.mean(list(prophet_scores.values()))
sc_s70 = prophet_scores.get(FOCAL_STORE, np.nan)
scores_all["Prophet"] = sc_all
scores_s70["Prophet"] = sc_s70
print(f"Prophet | all: {sc_all:.4f}  store {FOCAL_STORE}: {sc_s70:.4f}  ({len(prophet_scores)} stores fitted)")

# %% [markdown]
# ## Summary tables

# %%
SEP = "-" * 60

def print_table(scores, title):
    print(f"\n{title:^60}")
    print(SEP)
    print(f"{'Model':<40} {'RMSPE':>8}")
    print(SEP)
    for k, v in sorted(scores.items(), key=lambda x: x[1]):
        print(f"{k:<40} {v:>8.4f}")
    print(SEP)

print_table(scores_s70, f"Store {FOCAL_STORE} results")
print_table(scores_all, "All-stores results")

# %% Validation plot — store FOCAL_STORE, best model (XGBoost global)
val_s70 = val[val["Store"] == FOCAL_STORE].copy()
feat_s70 = X_vl_g[mask70]
pred_s70 = pred_xgb_g[mask70]

# Re-align dates (val_s70 may have closed days filtered out vs our tree val)
plot_df = val_s70[val_s70["Sales"] > 0].copy().reset_index(drop=True)
plot_df["pred_xgb"] = pred_s70[:len(plot_df)]

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for ax, col, label, colour in [
    (axes[0], "Sales",    "Actual",  "steelblue"),
    (axes[0], "pred_xgb","XGBoost", "coral"),
]:
    ax.plot(plot_df["Date"], plot_df[col], lw=1.2, label=label, color=colour,
            linestyle="-" if col == "Sales" else "--")
axes[0].set(title=f"Store {FOCAL_STORE} — actual vs XGBoost (RMSPE {sc_s70:.4f})",
            ylabel="Sales (€)")
axes[0].legend()

axes[1].bar(plot_df["Date"],
            (plot_df["pred_xgb"] - plot_df["Sales"]) / plot_df["Sales"] * 100,
            color="steelblue", width=1)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set(ylabel="% error", xlabel="Date")

plt.tight_layout()
plt.savefig(f"outputs/model_comparison_store{FOCAL_STORE}.png", bbox_inches="tight")
print(f"\nPlot saved: outputs/model_comparison_store{FOCAL_STORE}.png")
