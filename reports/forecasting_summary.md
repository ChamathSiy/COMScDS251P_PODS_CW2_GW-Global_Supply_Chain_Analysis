# Forecasting Summary - Global Supply Chain Risk & Logistics Analysis

## Scope
This report adds the forecasting stage to the existing ETL, EDA, and clustering workflow. Three operational time-series were forecast using the cleaned shipment dataset:

- **Order volume** (daily shipment count)
- **Average lead time** (daily mean lead time in days)
- **Disruption rate** (daily share of disrupted shipments)

## Modelling Approach
The following models were evaluated on the final 60 days of the dataset:

- Seasonal Naive baseline (order volume only)
- Holt-Winters Exponential Smoothing
- SARIMAX with operational exogenous drivers

The best model for each target was selected using **lowest RMSE**.

## Best Model Summary
| Target          | Model        |      MAE |     RMSE |    MAPE |   Accuracy |
|:----------------|:-------------|---------:|---------:|--------:|-----------:|
| avg_lead_time   | Holt-Winters | 5.08934  | 6.59456  | 45.6869 |    54.3131 |
| disruption_rate | Holt-Winters | 0.158049 | 0.198277 | 28.6516 |    71.3484 |
| order_volume    | Holt-Winters | 2.02906  | 2.56179  | 33.6167 |    66.3833 |

## Key Forecast Insights
- **Order volume:** recent 30-day average = **7.30** shipments/day, next 30-day forecast average = **7.13** shipments/day (**-2.35%** change).
- **Average lead time:** recent 30-day average = **14.78** days, next 30-day forecast average = **14.53** days (**-1.65%** change).
- **Disruption rate:** recent 30-day average = **0.609**, next 30-day forecast average = **0.608** (**-0.10%** change).

## Operational Interpretation
- Shipment demand appears **stable**, with only a slight short-term softening in daily order volume.
- Lead time is forecast to **improve marginally**, suggesting no immediate deterioration in delivery performance.
- Disruption exposure remains **persistent at around 0.61**, which indicates continued operational risk and the need for proactive mitigation.

## Files Added
- `notebooks/04_forecasting.ipynb`
- `scripts/forecasting_pipeline.py`
- `data/processed/daily_supply_chain_timeseries.csv`
- `data/processed/forecast_model_metrics.csv`
- `data/processed/*_test_predictions.csv`
- `data/processed/future_30_day_forecasts.csv`
- `images/Forecast - *.png`

## Recommendation
Use the forecasts in weekly planning meetings to monitor shipment demand, carrier performance, and disruption risk. The disruption-rate forecast in particular should be linked to contingency plans, carrier escalation, and route prioritisation.
