# California Property Close Price Prediction

Predicting residential close prices across California from CRMLS listing data, designed to value a property whether or not it's currently for sale.

## Overview

Most price-prediction models lean on listing-time signals (asking price, days on market) that only exist while a property is actively listed. This model is built for the harder, more general case: estimate a residential property's close price from its physical and locational characteristics alone, so it works for off-market properties too.

## Key design decisions

- **Leakage-aware feature set** - `ListPrice`, `DaysOnMarket`, and other listing-time-only fields are excluded. Including them would make the model unusable for off-market inference and would silently inflate offline accuracy.
- **Walk-forward validation** - trains on a rolling window of the 12 months immediately preceding the moving test month to reflect how the model would actually be retrained over time.
- **Scope** - restricted to single family residential listings.
- **Stacked ensemble over a single model** - the final model blends XGBoost, LightGBM, and CatBoost through a Ridge meta-learner rather than picking one winner. Each base model brings different strengths (e.g. CatBoost's native handling of high-cardinality categoricals), and letting Ridge learn the combination weights outperformed any single tuned model or a hand-picked blend ratio.
- **The Streamlit demo intentionally uses a simpler, separate model.** *In Progress*

## Dataset

- **Source**: CRMLS (California Regional Multiple Listing Service) sold-listing data
- **Size**: ~794K rows, 82 raw columns
- **Target**: `ClosePrice`

## Feature Engineering

Five engineered features made it into the final model, each validated individually before inclusion - tested as a single addition against a Random Forest baseline first, then confirmed on the actual gradient-boosting ensemble via SHAP:

- **`CompPriceKNN`** - mean close price of the 15 nearest geographic comps (raw lat/long), computed without any information from the row being predicted. By far the strongest single feature in the final model (SHAP-confirmed).
- **`BedBathRatio`** - bedrooms ÷ bathrooms, computed from unscaled raw values.
- **`PropertyAgeAtSale`** - sale year minus year built.
- **`SchoolDistrictSpatial`** - spatial join against CA school district boundaries, replacing the raw `HighSchoolDistrict` field with a cleaner, more consistent district assignment.
- **`FireHazardZone`** - spatial join against CA wildfire hazard severity zones (state and local responsibility area layers).

**Tested and rejected**: cyclical time encoding (a single-year training window doesn't span enough seasons to separate seasonal effects from the overall price trend), a per-square-foot variant of the comp feature (redundant with `CompPriceKNN`), and a raw amenity count (negligible signal). Kept as documented negative results rather than deleted, so they don't get accidentally retried later.

**Pruned**: three near-zero-importance features (`AttachedGarageYN`, `Level_MultiSplit`, `BasementYN`) were dropped after gain-based importance analysis on the tuned models showed removing them cost nothing. A threshold sweep found this exact cutoff - more aggressive pruning measurably hurt performance, so the line was drawn empirically rather than by guess.

**One finding worth flagging**: `FireHazardZone` looked like the second-strongest individual feature in early Random Forest testing, but SHAP analysis on the actual ensemble showed its real marginal contribution is small - `CompPriceKNN` already captures most of the same spatial signal implicitly, since nearby comp prices reflect local fire risk (and everything else about a location) without needing it labeled explicitly. A reminder that a feature's standalone predictive power and its marginal value inside a larger model aren't the same thing.

**In progress, not yet merged**: leakage-safe local price aggregates (ZIP/district-level trailing-window median price-per-sqft, 90th-percentile price, and sales count as a market-liquidity signal) plus a raw/log1p prediction blend, adapted from a collaborator's independently-validated approach. Early results are promising but haven't yet replaced the model reflected in the Results table below.

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
├── data/                       # gitignored - not committed; parquet handoffs between notebooks, model artifacts
└── README.md
```

## Tech stack

Python · pandas · scikit-learn · XGBoost · LightGBM · CatBoost · SHAP · GeoPandas · Streamlit · joblib
