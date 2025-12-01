#!/usr/bin/env python3
"""
pricing_model.py

Prototype pricing model for a natural gas storage contract.

Contents:
 - load_nat_gas_csv(path) -> pandas.DataFrame
 - build_price_estimator_from_df(df) -> (estimator_fn, metadata)
 - price_storage_contract(..., price_estimator=...) -> float

Usage:
 - Edit the CSV path or pass it when running.
 - This script is intentionally dependency-light: pandas + numpy only.
"""

from datetime import datetime
from typing import Callable, Tuple
import sys

import numpy as np
import pandas as pd


def load_nat_gas_csv(path: str) -> pd.DataFrame:
    """
    Load monthly natural gas CSV and parse Dates & Prices.

    Expected CSV columns: a date column (e.g. 'Dates') and a numeric price column (e.g. 'Prices').

    The function will:
      - try to read the CSV
      - find a date-like column (common names or first column)
      - parse dates with explicit format '%m/%d/%y' (falls back to flexible parser)
      - find a single numeric price column (or the first numeric if multiple)
      - return cleaned DataFrame sorted by date

    Raises:
      FileNotFoundError, ValueError on invalid structure.
    """
    # Read
    df = pd.read_csv(path, engine="python")

    # Identify date column (common names)
    date_candidates = [c for c in df.columns if c.lower() in ("date", "dates", "month", "dt")]
    if date_candidates:
        date_col = date_candidates[0]
    else:
        # fallback to first column if it looks date-like
        date_col = df.columns[0]

    # Parse dates: explicit format first (MM/DD/YY), then fallback
    try:
        df[date_col] = pd.to_datetime(df[date_col], format="%m/%d/%y", errors="coerce")
        if df[date_col].isna().any():
            # fallback to generic parser for rows that didn't parse
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)

    # Identify price column (common names)
    price_candidates = [c for c in df.columns if any(k in c.lower() for k in ("price", "prices", "spot", "value", "close"))]
    if price_candidates:
        price_col = price_candidates[0]
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found to use as price.")
        price_col = numeric_cols[0]

    # Clean and sort
    df = df[[date_col, price_col]].rename(columns={date_col: "Dates", price_col: "Prices"})
    df["Prices"] = pd.to_numeric(df["Prices"], errors="coerce")
    df = df.dropna(subset=["Dates", "Prices"]).sort_values("Dates").reset_index(drop=True)

    if df.empty:
        raise ValueError("Loaded DataFrame is empty after parsing dates and prices.")

    return df


def build_price_estimator_from_df(df: pd.DataFrame) -> Tuple[Callable[[datetime], float], dict]:
    """
    Build a simple price estimator from the observed monthly series.

    Model approach:
      - Fit a linear trend to time (Days since first observed date) using numpy.polyfit
      - Compute monthly seasonal offsets from residuals (month -> mean residual)
      - Estimator(date) = trend(date) + seasonal_offset(month)

    Returns:
      - estimate_price(date) callable
      - metadata dict with keys: t0, trend_coef, seasonal_lookup, last_observed_date, forecast_horizon_days
    """
    if "Dates" not in df.columns or "Prices" not in df.columns:
        raise ValueError("DataFrame must contain 'Dates' and 'Prices' columns.")

    df = df.copy()
    t0 = df["Dates"].min()
    df["t_days"] = (df["Dates"] - t0).dt.days.astype(float)

    # Linear trend (degree=1)
    # polyfit returns [slope, intercept]
    slope, intercept = np.polyfit(df["t_days"].values, df["Prices"].values, 1)

    # Seasonality: residual from trend, grouped by month
    df["trend"] = slope * df["t_days"] + intercept
    df["residual"] = df["Prices"] - df["trend"]
    df["month"] = df["Dates"].dt.month
    seasonal_means = df.groupby("month")["residual"].mean()
    # Center seasonal offsets so their mean is zero (keeps intercept as long term level)
    seasonal_means = seasonal_means - seasonal_means.mean()
    seasonal_lookup = seasonal_means.to_dict()

    last_observed = df["Dates"].max()
    # default forecast horizon 365 days beyond last observed
    forecast_horizon_days = 365

    def estimate_price(date_input) -> float:
        """
        Estimate price for a given date-like input (str or datetime-like).
        Raises ValueError if date cannot be parsed.
        """
        # robust parsing
        if isinstance(date_input, str):
            d = pd.to_datetime(date_input, errors="coerce")
        elif isinstance(date_input, (pd.Timestamp, datetime)):
            d = pd.to_datetime(date_input)
        else:
            d = pd.to_datetime(date_input, errors="coerce")

        if pd.isna(d):
            raise ValueError(f"Could not parse date_input: {date_input}")

        t = (d - t0).days
        trend_val = slope * t + intercept
        seasonal_val = seasonal_lookup.get(d.month, 0.0)
        return float(trend_val + seasonal_val)

    metadata = {
        "t0": t0,
        "trend_coef": (slope, intercept),
        "seasonal_lookup": seasonal_lookup,
        "last_observed_date": last_observed,
        "forecast_horizon_days": forecast_horizon_days,
    }

    return estimate_price, metadata


def price_storage_contract(
    inject_dates,
    withdraw_dates,
    inject_rate: float,
    withdraw_rate: float,
    max_volume: float,
    storage_cost_monthly: float,
    inject_fee: float = 0.0,
    withdraw_fee: float = 0.0,
    transport_fee: float = 0.0,
    price_estimator: Callable[[datetime], float] = None,
) -> float:
    """
    Price a storage contract.

    Parameters:
      - inject_dates: iterable of date-like (strings or datetime)
      - withdraw_dates: iterable of date-like
      - inject_rate: volume injected per injection event (MMBtu)
      - withdraw_rate: volume withdrawn per withdrawal event (MMBtu)
      - max_volume: maximum storage capacity (MMBtu)
      - storage_cost_monthly: monthly rental cost for stored gas (currency units)
      - inject_fee / withdraw_fee / transport_fee: event-based fees (currency units)
      - price_estimator: callable(date) -> price per MMBtu

    Returns:
      - total_value: float (positive means profitable for the storage buyer)
    """
    if price_estimator is None:
        raise ValueError("price_estimator callable must be provided.")

    # Normalize and sort dates
    inject_dates = sorted(pd.to_datetime(d) for d in inject_dates)
    withdraw_dates = sorted(pd.to_datetime(d) for d in withdraw_dates)

    # Basic sanity checks
    if not inject_dates:
        raise ValueError("At least one injection date is required.")
    if not withdraw_dates:
        raise ValueError("At least one withdrawal date is required.")

    tot_injected = inject_rate * len(inject_dates)
    tot_withdrawn = withdraw_rate * len(withdraw_dates)
    if tot_withdrawn > tot_injected:
        # It's allowed in some structures but here we enforce feasibility
        raise ValueError("Total withdrawal volume exceeds total injected volume (feasibility).")

    if inject_rate > max_volume:
        raise ValueError("Single injection volume exceeds max storage capacity.")

    current_volume = 0.0
    total_purchases = 0.0
    total_sales = 0.0
    total_fees = 0.0

    # Process injection events in chronological order (interleaving not handled here; assumes injects happen before corresponding withdraws if needed)
    for d in inject_dates:
        # check capacity
        if current_volume + inject_rate > max_volume + 1e-9:
            raise ValueError(f"Storage capacity would be exceeded on injection date {d.date()}.")
        price = float(price_estimator(d))
        total_purchases += inject_rate * price
        total_fees += inject_fee + transport_fee
        current_volume += inject_rate

    # Process withdrawal events
    for d in withdraw_dates:
        if current_volume - withdraw_rate < -1e-9:
            raise ValueError(f"Insufficient volume for withdrawal on {d.date()}.")
        price = float(price_estimator(d))
        total_sales += withdraw_rate * price
        total_fees += withdraw_fee + transport_fee
        current_volume -= withdraw_rate

    # Storage months: charge for months between first injection and last withdrawal inclusive
    start = min(inject_dates)
    end = max(withdraw_dates)
    # Use Period arithmetic to get whole month count inclusive
    months_diff = (end.to_period("M") - start.to_period("M")).n
    storage_months = int(months_diff) + 1 if months_diff >= 0 else 0
    total_storage_cost = storage_months * storage_cost_monthly

    # Final valuation
    total_value = total_sales - total_purchases - total_fees - total_storage_cost
    return float(total_value)


# Example usage when run as a script
if __name__ == "__main__":
    # Default CSV path (edit if needed)
    default_csv = r"C:\Users\HP\Desktop\Nat_Gas.csv"

    # Allow override: python pricing_model.py path/to/Nat_Gas.csv
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv

    # Load data and build estimator
    try:
        df_prices = load_nat_gas_csv(csv_path)
    except Exception as e:
        raise SystemExit(f"Failed to load CSV at {csv_path}: {e}")

    estimate_price_fn, meta = build_price_estimator_from_df(df_prices)

    # Print a small demonstration (safe, concise)
    sample_date = "2025-02-15"
    try:
        sample_price = estimate_price_fn(sample_date)
    except Exception as e:
        raise SystemExit(f"Failed to estimate price for {sample_date}: {e}")

    # Example contract
    example_value = price_storage_contract(
        inject_dates=["2024-06-01", "2024-07-01"],
        withdraw_dates=["2024-12-15", "2025-01-10"],
        inject_rate=500000.0,
        withdraw_rate=500000.0,
        max_volume=2_000_000.0,
        storage_cost_monthly=100000.0,
        inject_fee=10_000.0,
        withdraw_fee=10_000.0,
        transport_fee=50_000.0,
        price_estimator=estimate_price_fn,
    )

    print(f"Sample estimate for {sample_date}: {sample_price:.6f}")
    print(f"Example contract value (currency units): {example_value:.6f}")
