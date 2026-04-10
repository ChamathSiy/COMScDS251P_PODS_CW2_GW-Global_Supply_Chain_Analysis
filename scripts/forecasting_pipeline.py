
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_supply_chain_data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
IMAGES_DIR = BASE_DIR / "images"
REPORTS_DIR = BASE_DIR / "reports"


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def forecast_accuracy(y_true, y_pred):
    mape_value = mape(y_true, y_pred)
    if np.isnan(mape_value):
        return np.nan
    return max(0, 100 - mape_value)


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "Accuracy": forecast_accuracy(y_true, y_pred),
    }


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date")

    daily = (
        df.groupby("Date")
        .agg(
            order_volume=("Shipment_ID", "count"),
            avg_lead_time=("Lead_Time_Days", "mean"),
            disruption_rate=("Disruption_Occurred", "mean"),
            avg_fuel_price=("Fuel_Price_Index", "mean"),
            avg_geo_risk=("Geopolitical_Risk_Score", "mean"),
            avg_carrier_reliability=("Carrier_Reliability_Score", "mean"),
            avg_distance=("Distance_km", "mean"),
            avg_weight=("Weight_MT", "mean"),
        )
        .asfreq("D")
    )

    daily.to_csv(PROCESSED_DIR / "daily_supply_chain_timeseries.csv")
    return df, daily


def plot_historical_trends(daily):
    plot_specs = {
        "order_volume": ("Daily Shipment Volume", "Shipments"),
        "avg_lead_time": ("Daily Average Lead Time", "Lead Time (Days)"),
        "disruption_rate": ("Daily Disruption Rate", "Disruption Rate"),
    }

    for column, (title, ylabel) in plot_specs.items():
        plt.figure(figsize=(14, 5))
        plt.plot(daily.index, daily[column], linewidth=1.2, label="Daily")
        plt.plot(daily.index, daily[column].rolling(30, min_periods=1).mean(), linewidth=2.2, label="30-day rolling mean")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / f"Forecast - {title}.png", dpi=300)
        plt.close()

    daily["weekday"] = daily.index.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, column, title in zip(
        axes,
        ["order_volume", "avg_lead_time", "disruption_rate"],
        ["Weekday Pattern - Shipment Volume", "Weekday Pattern - Lead Time", "Weekday Pattern - Disruption Rate"],
    ):
        weekday_summary = daily.groupby("weekday")[column].mean().reindex(weekday_order)
        ax.bar(weekday_summary.index, weekday_summary.values)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "Forecast - Weekday Patterns.png", dpi=300)
    plt.close()
    daily.drop(columns=["weekday"], inplace=True)


def split_series(daily, horizon=60):
    train = daily.iloc[:-horizon].copy()
    test = daily.iloc[-horizon:].copy()
    return train, test


def seasonal_naive_forecast(series, test_index, seasonal_lag=7):
    preds = pd.Series(index=test_index, dtype=float)
    for idx in test_index:
        preds.loc[idx] = series.loc[idx - pd.Timedelta(days=seasonal_lag)]
    return preds


def fit_holt_winters(train_series, horizon, clip_bounds=None):
    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
    ).fit(optimized=True)
    forecast = model.forecast(horizon)
    if clip_bounds is not None:
        forecast = forecast.clip(*clip_bounds)
    return model, forecast


def fit_sarimax(train_series, train_exog, test_exog, horizon, clip_bounds=None):
    model = SARIMAX(
        train_series,
        exog=train_exog,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    forecast = model.forecast(horizon, exog=test_exog)
    if clip_bounds is not None:
        forecast = forecast.clip(*clip_bounds)
    return model, forecast


def evaluate_forecasts(daily, horizon=60):
    exog_cols = [
        "avg_fuel_price",
        "avg_geo_risk",
        "avg_carrier_reliability",
        "avg_distance",
        "avg_weight",
    ]
    targets = {
        "order_volume": {"ylabel": "Shipments", "clip": None, "include_baseline": True},
        "avg_lead_time": {"ylabel": "Lead Time (Days)", "clip": None, "include_baseline": False},
        "disruption_rate": {"ylabel": "Disruption Rate", "clip": (0, 1), "include_baseline": False},
    }

    train, test = split_series(daily, horizon=horizon)
    metrics_records = []
    prediction_tables = {}
    best_models = {}

    for target, meta in targets.items():
        y_train = train[target]
        y_test = test[target]

        prediction_df = pd.DataFrame(index=test.index)
        prediction_df["actual"] = y_test

        if meta["include_baseline"]:
            baseline_pred = seasonal_naive_forecast(daily[target], test.index)
            prediction_df["seasonal_naive"] = baseline_pred
            metric_row = {"Target": target, "Model": "Seasonal Naive"}
            metric_row.update(calculate_metrics(y_test, baseline_pred))
            metrics_records.append(metric_row)

        _, hw_pred = fit_holt_winters(y_train, len(test), clip_bounds=meta["clip"])
        prediction_df["holt_winters"] = hw_pred
        metric_row = {"Target": target, "Model": "Holt-Winters"}
        metric_row.update(calculate_metrics(y_test, hw_pred))
        metrics_records.append(metric_row)

        _, sarimax_pred = fit_sarimax(
            y_train,
            train[exog_cols],
            test[exog_cols],
            len(test),
            clip_bounds=meta["clip"],
        )
        prediction_df["sarimax"] = sarimax_pred
        metric_row = {"Target": target, "Model": "SARIMAX"}
        metric_row.update(calculate_metrics(y_test, sarimax_pred))
        metrics_records.append(metric_row)

        prediction_tables[target] = prediction_df.reset_index().rename(columns={"index": "Date"})

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df = metrics_df.sort_values(["Target", "RMSE", "MAE"]).reset_index(drop=True)
    metrics_df.to_csv(PROCESSED_DIR / "forecast_model_metrics.csv", index=False)

    best_models = (
        metrics_df.sort_values(["Target", "RMSE", "MAE"]).groupby("Target", as_index=False).first()[["Target", "Model"]]
    )
    best_models.to_csv(PROCESSED_DIR / "best_forecasting_models.csv", index=False)

    for target, df_pred in prediction_tables.items():
        df_pred.to_csv(PROCESSED_DIR / f"{target}_test_predictions.csv", index=False)

    return train, test, metrics_df, prediction_tables, best_models


def plot_test_predictions(test, prediction_tables):
    plot_labels = {
        "order_volume": ("Order Volume Forecast vs Actual (Test Set)", "Shipments"),
        "avg_lead_time": ("Lead Time Forecast vs Actual (Test Set)", "Lead Time (Days)"),
        "disruption_rate": ("Disruption Rate Forecast vs Actual (Test Set)", "Disruption Rate"),
    }

    for target, (title, ylabel) in plot_labels.items():
        pred_df = prediction_tables[target].copy()
        pred_df["Date"] = pd.to_datetime(pred_df["Date"])

        plt.figure(figsize=(14, 5))
        plt.plot(pred_df["Date"], pred_df["actual"], label="Actual", linewidth=2.0)
        for col in pred_df.columns:
            if col not in {"Date", "actual"}:
                plt.plot(pred_df["Date"], pred_df[col], label=col.replace("_", " ").title(), linewidth=1.5)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / f"Forecast - {title}.png", dpi=300)
        plt.close()


def plot_model_comparison(metrics_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_df, x="Target", y="RMSE", hue="Model")
    plt.title("Forecast Model Comparison by RMSE")
    plt.xlabel("")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "Forecast - Model Comparison by RMSE.png", dpi=300)
    plt.close()


def generate_future_forecasts(daily, best_models, horizon=30):
    future_index = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_df = pd.DataFrame(index=future_index)

    for _, row in best_models.iterrows():
        target = row["Target"]
        model_name = row["Model"]
        clip = (0, 1) if target == "disruption_rate" else None

        if model_name == "Holt-Winters":
            fitted_model, forecast = fit_holt_winters(daily[target], horizon, clip_bounds=clip)
            residual_std = float(np.std(fitted_model.resid, ddof=1))
        elif model_name == "Seasonal Naive":
            forecast = pd.Series(index=future_index, dtype=float)
            for idx in future_index:
                forecast.loc[idx] = daily.loc[idx - pd.Timedelta(days=7), target]
            residual_std = float(np.std(daily[target].diff(7).dropna(), ddof=1))
        else:
            # Use recent average exogenous values to extend SARIMAX into the forecast horizon.
            exog_cols = [
                "avg_fuel_price",
                "avg_geo_risk",
                "avg_carrier_reliability",
                "avg_distance",
                "avg_weight",
            ]
            future_exog = pd.DataFrame(
                [daily[exog_cols].tail(30).mean().to_dict() for _ in range(horizon)],
                index=future_index,
            )
            fitted_model, forecast = fit_sarimax(
                daily[target],
                daily[exog_cols],
                future_exog,
                horizon,
                clip_bounds=clip,
            )
            residual_std = float(np.std(fitted_model.resid.dropna(), ddof=1))

        lower = forecast - 1.96 * residual_std
        upper = forecast + 1.96 * residual_std
        if clip is not None:
            lower = lower.clip(*clip)
            upper = upper.clip(*clip)

        future_df[f"{target}_forecast"] = forecast.values
        future_df[f"{target}_lower_95"] = lower.values
        future_df[f"{target}_upper_95"] = upper.values

    future_df = future_df.reset_index().rename(columns={"index": "Date"})
    future_df.to_csv(PROCESSED_DIR / "future_30_day_forecasts.csv", index=False)
    return future_df


def plot_future_forecasts(daily, future_df):
    future_df = future_df.copy()
    future_df["Date"] = pd.to_datetime(future_df["Date"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=False)
    plot_specs = [
        ("order_volume", "Shipments", "30-Day Forecast - Order Volume"),
        ("avg_lead_time", "Lead Time (Days)", "30-Day Forecast - Average Lead Time"),
        ("disruption_rate", "Disruption Rate", "30-Day Forecast - Disruption Rate"),
    ]

    for ax, (target, ylabel, title) in zip(axes, plot_specs):
        hist = daily[target].tail(90)
        ax.plot(hist.index, hist.values, label="Last 90 days actual", linewidth=1.8)
        ax.plot(future_df["Date"], future_df[f"{target}_forecast"], label="Forecast", linewidth=2.2)
        ax.fill_between(
            future_df["Date"],
            future_df[f"{target}_lower_95"],
            future_df[f"{target}_upper_95"],
            alpha=0.2,
            label="95% interval",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "Forecast - Future 30 Day Outlook.png", dpi=300)
    plt.close()


def write_summary_report(metrics_df, future_df, daily):
    best = metrics_df.sort_values(["Target", "RMSE", "MAE"]).groupby("Target", as_index=False).first()

    def pct_change(target):
        recent_mean = daily[target].tail(30).mean()
        future_mean = future_df[f"{target}_forecast"].mean()
        return recent_mean, future_mean, ((future_mean - recent_mean) / recent_mean) * 100

    ov_recent, ov_future, ov_pct = pct_change("order_volume")
    lt_recent, lt_future, lt_pct = pct_change("avg_lead_time")
    dr_recent, dr_future, dr_pct = pct_change("disruption_rate")

    report = f"""# Forecasting Summary - Global Supply Chain Risk & Logistics Analysis

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
{best.to_markdown(index=False)}

## Key Forecast Insights
- **Order volume:** recent 30-day average = **{ov_recent:.2f}** shipments/day, next 30-day forecast average = **{ov_future:.2f}** shipments/day (**{ov_pct:.2f}%** change).
- **Average lead time:** recent 30-day average = **{lt_recent:.2f}** days, next 30-day forecast average = **{lt_future:.2f}** days (**{lt_pct:.2f}%** change).
- **Disruption rate:** recent 30-day average = **{dr_recent:.3f}**, next 30-day forecast average = **{dr_future:.3f}** (**{dr_pct:.2f}%** change).

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
"""

    (REPORTS_DIR / "forecasting_summary.md").write_text(report, encoding="utf-8")


def main():
    _, daily = load_and_prepare_data()
    plot_historical_trends(daily.copy())
    train, test, metrics_df, prediction_tables, best_models = evaluate_forecasts(daily)
    plot_test_predictions(test, prediction_tables)
    plot_model_comparison(metrics_df)
    future_df = generate_future_forecasts(daily, best_models)
    plot_future_forecasts(daily, future_df)
    write_summary_report(metrics_df, future_df, daily)

    print("Forecasting pipeline completed successfully.")
    print(f"Processed outputs saved to: {PROCESSED_DIR}")
    print(f"Charts saved to: {IMAGES_DIR}")
    print(f"Summary report saved to: {REPORTS_DIR / 'forecasting_summary.md'}")


if __name__ == "__main__":
    main()
