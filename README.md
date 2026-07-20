# California property close price prediction

Predicting residential close prices across California from CRMLS listing data — designed to value a property whether or not it's currently for sale.

## Overview

Most price-prediction models lean on listing-time signals (asking price, days on market) that only exist while a property is actively listed. This model is built for the harder, more general case: estimate a residential property's close price from its physical and locational characteristics alone, so it works for off-market properties too.

## Key design decisions

- **Leakage-aware feature set** — `ListPrice`, `DaysOnMarket`, and other listing-time-only fields are excluded. Including them would make the model unusable for off-market inference and would silently inflate offline accuracy.
- **Walk-forward validation** — trains on a rolling window of the 12 months immediately preceding the moving test month to reflect how the model would actually be retrained over time.
- **Scope** — restricted to single family residential listings
- **Feature engineering** — bed/bath ratio, property age, and a spatial join against CA school district boundaries (2024–25) as a richer geographic signal than raw coordinates.

## Dataset

- **Source**: CRMLS (California Regional Multiple Listing Service) sold-listing data
- **Size**: ~794K rows, 82 raw columns
- **Target**: `ClosePrice`

## Project status

In progress — currently completing tree-based model comparisons (baseline vs. Random Forest). Advanced boosting, expanded evaluation, and the serving/deployment layer below are planned next.

- [x] Data exploration & EDA
- [x] Preprocessing pipeline (leakage-aware, walk-forward split)
- [x] Baseline model (Linear Regression)
- [ ] Tree-based model comparison (Decision Tree, Random Forest) — *in progress*
- [ ] Feature engineering (bed/bath ratio, property age, school district join)
- [ ] Advanced models (XGBoost / LightGBM) with experiment tracking
- [ ] Expanded evaluation (MAPE, MdAPE, price-band analysis)
- [ ] Serving API (FastAPI + Docker) and Streamlit UI
- [ ] CI/CD pipeline and drift monitoring

## Results

*Populated as each modeling stage completes.*

| Model | R² | MAPE | RMSE |
|---|---|---|---|
| Linear Regression (baseline) | 0.5188 | 0.3668 | $550,330 |
| Decision Tree | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |

## Repository structure

```
crmls-price-prediction/
├── .github/workflows/ci.yml    # test + build pipeline
├── notebooks/                  # exploration, preprocessing, model dev
├── src/                        # shared preprocessing/feature code,
│                                #   imported by both notebooks and the API
├── serving/                    # FastAPI app + Dockerfile
├── ui/                         # Streamlit frontend
├── tests/                      # pytest suite run by CI
├── monitoring/                 # drift-check script vs. training baseline
├── data/                       # gitignored — not committed
└── README.md
```

## Running it locally

```bash
# train / explore
jupyter lab notebooks/

# serve the model
docker build -t crmls-api ./serving
docker run -p 8000:8000 crmls-api

# run the UI
streamlit run ui/streamlit_app.py
```

## Tech stack

Python · pandas · scikit-learn · XGBoost · MLflow · FastAPI · Docker · GitHub Actions · Streamlit
