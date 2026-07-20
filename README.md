# California Property Close Price Prediction

Predicting residential close prices across California from CRMLS listing data, designed to value a property whether or not it's currently for sale.

## Overview

Most price-prediction models lean on listing-time signals (asking price, days on market) that only exist while a property is actively listed. This model is built for the harder, more general case: estimate a residential property's close price from its physical and locational characteristics alone, so it works for off-market properties too.

## Key design decisions

- **Leakage-aware feature set** — `ListPrice`, `DaysOnMarket`, and other listing-time-only fields are excluded. Including them would make the model unusable for off-market inference and would silently inflate offline accuracy.
- **Walk-forward validation** — trains on a rolling window of the 12 months immediately preceding the moving test month to reflect how the model would actually be retrained over time.
- **Scope** — restricted to single family residential listings

## Dataset

- **Source**: CRMLS (California Regional Multiple Listing Service) sold-listing data
- **Size**: ~794K rows, 82 raw columns
- **Target**: `ClosePrice`

## Project status

- [x] Data exploration & EDA
- [x] Preprocessing pipeline (leakage-aware, walk-forward split)
- [x] Baseline model (Linear Regression)
- [X] Tree-based model comparison (Decision Tree, Random Forest) — *in progress*
- [ ] Feature engineering (bed/bath ratio, property age, school district join)
- [ ] Advanced models (XGBoost / LightGBM) with experiment tracking
- [ ] Expanded evaluation (MAPE, MdAPE, price-band analysis)
- [ ] Serving API (FastAPI + Docker) and Streamlit UI
- [ ] CI/CD pipeline and drift monitoring

## Results

*Populated as each modeling stage completes.*

| Model | R² | MAPE | RMSE |
|---|---|---|---|
| Linear Regression (baseline) | 0.8302 | 0.2054 | $363,642 |
| Decision Tree | 0.7962 | 0.1718 | $398,428 |
| Random Forest | 0.8747 | 0.1211 | $312,482 |
| CatBoost | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |

## Repository structure

```
crmls-price-prediction/
├── notebooks/                  # 
├── data/                       # gitignored — not committed
└── README.md
```

## Tech stack

Python · pandas · scikit-learn 
