# Rossmann Sales Forecasting & Causal Analysis

End-to-end data science project on the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) Kaggle dataset: 1,115 German drugstores, daily sales from 2013–2015.

Two objectives:

1. **Forecasting** — benchmark a full model matrix (OLS variants, Random Forest, XGBoost, LightGBM, CatBoost, MLP, Prophet) on RMSPE, the Kaggle competition metric.
2. **Causal inference** — quantify the effect of nearby competitor openings on Rossmann sales using Regression Discontinuity in Time (RDiT), with distance heterogeneity and a placebo test.

---

## Results at a glance

### Forecasting — all-stores RMSPE (validation: May–Jul 2015)

| Model | RMSPE | Notes |
|---|---|---|
| Seasonal naive | 0.1767 | Baseline: same day 52 weeks prior |
| OLS structural (linear) | 0.2302 | Calendar + promo, raw sales target |
| OLS structural (mult) | 0.2095 | Calendar + promo, log(Sales) target |
| MLP | 0.1835 | 2-layer NN, same features as tree models |
| OLS predictive (linear) | 0.1607 | Lag features replace store FE |
| OLS predictive (mult) | 0.1525 | Log-lags + log(Sales) target |
| Random Forest | 0.1455 | 200 trees, log(Sales) target |
| **LightGBM** | **0.1315** | **Best — histogram boosting, all features** |
| CatBoost | 0.1375 | Comparable to LightGBM |
| Prophet *(sample)* | *~0.1498* | *20-store sample; no lag features* |

### Store 70 — single-store models vs global models (filtered)

| Model | Single-store | Global (filtered) |
|---|---|---|
| OLS structural (mult) | 0.1270 | 0.1235 |
| OLS predictive (mult) | 0.1252 | 0.1267 |
| Random Forest | 0.1116 | 0.1184 |
| XGBoost | 0.1111 | — |
| LightGBM | — | **0.0982** |
| CatBoost | — | 0.1027 |

### Competition opening causal effect (RDiT)

| | Estimate | 95% CI | % change |
|---|---|---|---|
| Immediate sales impact (β) | −0.030 | [−0.041, −0.019] | **−2.9%** |
| Close competitors (<1.15 km) | −0.045 | | **−4.4%** |
| Medium distance | −0.044 | | **−4.3%** |
| Far competitors | +0.001 | | n.s. |
| Placebo (6 months early) | +0.004 | p = 0.57 | Pre-trends clean ✓ |

---

## Methodology

### Feature engineering (`03_feature_engineering.py`)

- **Date features**: year, month, week-of-year, day-of-month
- **Christmas/New Year dummies**: one binary per calendar date for Dec 15–24, Dec 26–31, Jan 2 — model learns each day's multiplier independently
- **Calendar flags**: `is_pre_holiday` (stockpiling day before closure), `is_bridge_day` (weekday between holiday and weekend), `is_month_end` (payday effect)
- **Promo2_active**: binary, whether Promo2 is active in that store's current month
- **Competition features**: `CompetitionDistance` (NaN → max observed + `no_competitor` flag), `months_since_competitor_opened`, plus **`comp_opened_last_6m`** and **`days_since_comp_opened`** to signal the window where lag features are unreliable (the lag looks back to pre-competition actuals)
- **Lag and rolling features** (computed on open days only, per store):
  - `sales_lag_7`, `sales_lag_14`: direct lag
  - `sales_lag_same_cond`: previous occurrence with same DayOfWeek × Promo — the strongest single predictor
  - `sales_roll4_same_cond`, `sales_roll8_same_cond`: rolling means over same-condition occurrences
  - `store_trend_56`: 56-day rolling mean of all open days (general level)

All lag features are computed on the full dataset sorted by date before the train/val split, so test-period rows look back into training actuals without leakage.

**Splits** (time-based only — no random shuffling):

| Split | Period | Purpose |
|---|---|---|
| `train` | 2013-01-01 → 2015-04-30 | Model fitting |
| `val` | 2015-05-01 → 2015-07-31 | Out-of-sample evaluation |
| `trainval` | 2013-01-01 → 2015-07-31 | Final refit before Kaggle submission |

### Model matrix (`04_models.py`)

Models are organised along two axes:

|  | Structural | Predictive |
|---|---|---|
| **Multiplicative** (log target) | OLS-SM | OLS-PM |
| **Linear** (raw target) | OLS-SL | OLS-PL |

**Structural** models use calendar and promo features; store fixed effects absorb the store-level baseline, leaving coefficients as clean seasonal multipliers or additive increments.

**Predictive** models replace store fixed effects and the Promo dummy with lag/rolling features. The same-condition lag (`sales_lag_same_cond`) groups by Store × DayOfWeek × Promo, so Promo status is already encoded — adding it separately would be redundant.

**Within-store demeaning** (Frisch-Waugh theorem) is used instead of 1,114 store dummy columns for the global OLS fits, avoiding the ~7 GB dense matrix that would otherwise be needed.

**Global tree models** (RF, XGBoost, LightGBM, CatBoost, MLP) use a shared feature set across all stores. Store identity is not included as a feature — the lag and rolling variables implicitly encode each store's sales level, making the model generalisable to unseen stores.

### Competition opening analysis (`05_competition_opening.py`)

**Identification strategy — Regression Discontinuity in Time (RDiT)**

For each of 163 "treated" stores (competitor opened during the training window with ≥60 days on each side), we estimate:

```
log(Sales_it) = α + β·POST_it + γ·t + δ·POST_it·t + ε_it
```

where `t` = days relative to competitor opening and `POST` = 1 after opening. `β` captures the immediate level shift. Store fixed effects are absorbed by within-store demeaning. Bandwidth: ±90 days.

**Placebo test**: re-run with a fake opening date 6 months earlier. β = +0.004, p = 0.57 — no pre-trend, confirming the result is not driven by pre-existing sales trajectories.

**Distance heterogeneity**: the effect is concentrated among stores with close (<1.15 km) or medium-distance competitors; far competitors have no measurable effect.

---

## Charts

| File | Description |
|---|---|
| `outputs/rmspe_comparison.png` | RMSPE league table — all models, both scopes |
| `outputs/actual_vs_pred_store70.png` | Actual vs LightGBM, store 70, full validation period |
| `outputs/shap_importance.png` | LightGBM SHAP feature importance (top 20) |
| `outputs/error_distribution.png` | % error distribution: naive vs OLS vs LightGBM |
| `outputs/comp_event_study.png` | Event study: monthly log sales around competitor opening |
| `outputs/comp_rdit.png` | RDiT scatter + fitted lines (weekly bins, ±90 days) |
| `outputs/comp_het_distance.png` | RDiT effect by competition distance tercile |

---

## Repository structure

```
notebooks/
  01_eda.py                  # Sales distributions, seasonality, store variation
  02_eda.py                  # Promo and holiday deep-dive
  03_feature_engineering.py  # Build train/val/trainval feature parquets
  04_models.py               # Full model benchmark (OLS → LightGBM → Prophet)
  05_competition_opening.py  # RDiT causal analysis of competitor openings
  06_model_comparison.py     # Chart generation (runs LightGBM + SHAP)
data/
  raw/      # Kaggle CSVs (gitignored — see download instructions below)
  processed/# Engineered features (gitignored)
outputs/    # All saved charts
src/        # Shared utilities
```

---

## Reproducing results

### 1. Get the data

```bash
# Option A — Kaggle CLI
pip install kaggle
kaggle competitions download -c rossmann-store-sales -p data/raw/
unzip data/raw/rossmann-store-sales.zip -d data/raw/

# Option B — manual
# Download from https://www.kaggle.com/c/rossmann-store-sales/data
# and place train.csv and store.csv in data/raw/
```

### 2. Run in Docker (recommended)

```bash
docker compose up -d
docker exec rossmann-sales-forecast-notebook-1 python notebooks/03_feature_engineering.py
docker exec rossmann-sales-forecast-notebook-1 python notebooks/04_models.py
docker exec rossmann-sales-forecast-notebook-1 python notebooks/05_competition_opening.py
docker exec rossmann-sales-forecast-notebook-1 python notebooks/06_model_comparison.py
```

### 3. Run locally

```bash
pip install -r requirements.txt
python notebooks/03_feature_engineering.py
python notebooks/04_models.py
python notebooks/05_competition_opening.py
python notebooks/06_model_comparison.py
```

**Runtime**: feature engineering ~2 min; `04_models.py` ~15–20 min (LightGBM + CatBoost dominate); `06_model_comparison.py` ~3 min (re-fits LightGBM + SHAP).

---

## Key findings

1. **Lag features dominate**: `sales_lag_same_cond` (previous occurrence with same weekday and promo status) is the strongest predictor — a store's own recent history predicts far better than any calendar or store-metadata feature alone.

2. **Log target matters**: multiplicative (log) OLS outperforms linear OLS on RMSPE by 3–6 pp because RMSPE penalises relative errors; large-volume stores dominate a linear model's loss.

3. **Structural OLS underperforms predictive OLS globally** (0.21 vs 0.15): the structural model must absorb store-level heterogeneity through demeaning and has no access to the recent trajectory. Within a single store the gap narrows considerably.

4. **Gradient boosting is king**: LightGBM (0.1315) and CatBoost (0.1375) beat all OLS variants and Random Forest, with no meaningful feature engineering advantage — the same features that power OLS predictive drive the tree models.

5. **Competition opening reduces sales by ~2.9%**: the effect is sharply localised to the opening month (RDiT bandwidth ±90 days), concentrated among close-distance competitors, and absent in the placebo test — consistent with a genuine causal effect rather than a spurious trend.

6. **`comp_opened_last_6m` / `days_since_comp_opened`**: these features signal to ML models that the lag features are unreliable during the 6-month window after a competitor opens (when the lag still reflects pre-competition sales). Including them gave a measurable lift to LightGBM on those stores.
