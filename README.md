# California Property Close Price Prediction

Predicting residential close prices across California from CRMLS listing data, designed to value a property whether or not it's currently for sale.

## Overview

Most price-prediction models lean on listing-time signals (asking price, days on market) that only exist while a property is actively listed. This model is built for the harder, more general case: estimate a residential property's close price from its physical and locational characteristics alone, so it works for off-market properties too.

## Key design decisions

- **Leakage-aware feature set** — `ListPrice`, `DaysOnMarket`, and other listing-time-only fields are excluded. Including them would make the model unusable for off-market inference and would silently inflate offline accuracy.
- **Walk-forward validation** — trains on a rolling window of the 12 months immediately preceding the moving test month to reflect how the model would actually be retrained over time.
- **Scope** — restricted to single family residential listings.
- **Stacked ensemble over a single model** — the final model blends XGBoost, LightGBM, and CatBoost through a Ridge meta-learner rather than picking one winner. Each base model brings different strengths (e.g. CatBoost's native handling of high-cardinality categoricals), and letting Ridge learn the combination weights outperformed any single tuned model or a hand-picked blend ratio.
- **The Streamlit demo intentionally uses a simpler, separate model.** The interactive UI takes four raw inputs (living area, beds, baths, lot size) and serves predictions from a standalone `RandomForestRegressor` trained directly on those columns, rather than the full stacked ensemble. The ensemble's strongest signal (`CompPriceKNN`, spatial joins, engineered ratios) isn't derivable from four raw inputs alone, so reusing it in the demo would mean silently falling back to defaults for most of what makes it accurate — a smaller model trained for exactly the inputs available is a more honest match for what the demo can actually know.

## Dataset

- **Source**: CRMLS (California Regional Multiple Listing Service) sold-listing data
- **Size**: ~794K rows, 82 raw columns
- **Target**: `ClosePrice`

## Results

*Populated as each modeling stage completes.*

| Model | R² | MAPE | RMSE |
|---|---|---|---|
| Linear Regression (baseline) | 0.5172 | 0.3687 | $547,680 |
| Linear Regression (Final) | 0.8630 | 0.1698 | $291,679 |
| Decision Tree | 0.8552 | 0.1340 | $299,919 |
| Random Forest | 0.8850 | 0.1129 | $267,299 |
| CatBoost | 0.9066 | 0.1049 | $240,837 |
| LightGBM | 0.9102 | 0.1022 | $236,168 |
| XGBoost | 0.9101 | 0.1031 | $236,303 |
| XGB+LGB+CatBoost stacked (Ridge) | 0.9145 | 0.1010 | $230,496 |

## Repository structure

```
crmls-price-prediction/
├── notebooks/                  # numbered analysis notebooks (01–08), run in order
├── app/                        # Streamlit demo (app.py, train_model.py)
├── data/                       # gitignored — not committed; parquet handoffs between notebooks, model artifacts
└── README.md
```

## Tech stack

Python · pandas · scikit-learn · XGBoost · LightGBM · CatBoost · SHAP · GeoPandas · Streamlit · joblib
