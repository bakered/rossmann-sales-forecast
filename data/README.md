# Data

The raw Rossmann data is not committed to this repository (Kaggle terms of use).

## Download

**Option 1 — Kaggle CLI:**
```bash
kaggle competitions download -c rossmann-store-sales -p data/raw/
cd data/raw && unzip rossmann-store-sales.zip
```

**Option 2 — Browser:**
Go to https://www.kaggle.com/c/rossmann-store-sales/data and download:
- `train.csv` (~40 MB) — daily sales for 1,115 stores, 2013–2015
- `test.csv` — the competition holdout set
- `store.csv` — store-level metadata (StoreType, Assortment, CompetitionDistance, PromoInterval)

Place all files in `data/raw/`.

## Dataset summary

| File | Rows | Key columns |
|------|------|-------------|
| train.csv | ~1 M | Store, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday |
| test.csv | ~41 k | Store, Date, Open, Promo, StateHoliday, SchoolHoliday |
| store.csv | 1,115 | Store, StoreType, Assortment, CompetitionDistance, PromoInterval |
