# -----------------------------------------------------------
# JP Morgan — Natural Gas Storage Contract Pricing
# Task 1: Estimate natural gas price for any date
# Method: Linear Trend + Monthly Seasonality
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    # Parse dates
    df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")
    df["Prices"] = pd.to_numeric(df["Prices"], errors="coerce")

    # Clean and sort
    df = df.dropna().sort_values("Dates").reset_index(drop=True)

    return df


def build_price_model(csv_path, forecast_years=1):
    df = load_and_prepare(csv_path)

    # Base date (start of time index)
    t0 = df["Dates"].min()

    # Numerical time index
    df["t_days"] = (df["Dates"] - t0).dt.days.astype(float)

    # -----------------------------
    # 1. Linear Trend Component
    # -----------------------------
    a, b = np.polyfit(df["t_days"], df["Prices"], 1)

    df["trend"] = a * df["t_days"] + b

    # -----------------------------
    # 2. Monthly Seasonality
    # -----------------------------
    df["residual"] = df["Prices"] - df["trend"]
    df["month"] = df["Dates"].dt.month

    # Seasonal adjustment per month
    seasonal_means = df.groupby("month")["residual"].mean()
    seasonal_means = seasonal_means - seasonal_means.mean()
    seasonal_lookup = seasonal_means.to_dict()

    # Forecast horizon (max allowed future date)
    last_date = df["Dates"].max()
    max_forecast_date = last_date + pd.DateOffset(years=forecast_years)

    # -------------------------------------------------------
    # Estimator function returned to user/tester
    # -------------------------------------------------------
    def estimate_price(date_input):
        """Return price estimate for any given date."""

        # Parse input date
        d = pd.to_datetime(date_input, errors="coerce")
        if pd.isna(d):
            raise ValueError("Invalid date format.")

        if d > max_forecast_date:
            raise ValueError(
                f"Date beyond forecast horizon. Use a date on or before {max_forecast_date.date()}."
            )

        # Trend
        t_days = (d - t0).days
        trend_val = a * t_days + b

        # Seasonality
        seasonal_val = seasonal_lookup.get(d.month, 0.0)

        return float(trend_val + seasonal_val)

    return estimate_price


# -----------------------------------------------------------
# Example usage (comment out for submission)
# -----------------------------------------------------------
# price_fn = build_price_model("Nat_Gas.csv")
# print(price_fn("2025-03-15"))
# -----------------------------------------------------------
# JP Morgan — Natural Gas Storage Contract Pricing
# Task 1: Estimate natural gas price for any date
# Method: Linear Trend + Monthly Seasonality
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    # Parse dates
    df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")
    df["Prices"] = pd.to_numeric(df["Prices"], errors="coerce")

    # Clean and sort
    df = df.dropna().sort_values("Dates").reset_index(drop=True)

    return df


def build_price_model(csv_path, forecast_years=1):
    df = load_and_prepare(csv_path)

    # Base date (start of time index)
    t0 = df["Dates"].min()

    # Numerical time index
    df["t_days"] = (df["Dates"] - t0).dt.days.astype(float)

    # -----------------------------
    # 1. Linear Trend Component
    # -----------------------------
    a, b = np.polyfit(df["t_days"], df["Prices"], 1)

    df["trend"] = a * df["t_days"] + b

    # -----------------------------
    # 2. Monthly Seasonality
    # -----------------------------
    df["residual"] = df["Prices"] - df["trend"]
    df["month"] = df["Dates"].dt.month

    # Seasonal adjustment per month
    seasonal_means = df.groupby("month")["residual"].mean()
    seasonal_means = seasonal_means - seasonal_means.mean()
    seasonal_lookup = seasonal_means.to_dict()

    # Forecast horizon (max allowed future date)
    last_date = df["Dates"].max()
    max_forecast_date = last_date + pd.DateOffset(years=forecast_years)

    # -------------------------------------------------------
    # Estimator function returned to user/tester
    # -------------------------------------------------------
    def estimate_price(date_input):
        """Return price estimate for any given date."""

        # Parse input date
        d = pd.to_datetime(date_input, errors="coerce")
        if pd.isna(d):
            raise ValueError("Invalid date format.")

        if d > max_forecast_date:
            raise ValueError(
                f"Date beyond forecast horizon. Use a date on or before {max_forecast_date.date()}."
            )

        # Trend
        t_days = (d - t0).days
        trend_val = a * t_days + b

        # Seasonality
        seasonal_val = seasonal_lookup.get(d.month, 0.0)

        return float(trend_val + seasonal_val)

    return estimate_price


# -----------------------------------------------------------
# Example usage (comment out for submission)
# -----------------------------------------------------------
# price_fn = build_price_model("Nat_Gas.csv")
# print(price_fn("2025-03-15"))
