"""
afrikana.utils
==============
Shared utility functions used across the afrikana-analytics package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def minmax_normalise(series: pd.Series) -> pd.Series:
    """Normalise a Series to [0, 1]. Returns 0.5 everywhere if range is zero."""
    r = series.max() - series.min()
    return (series - series.min()) / r if r > 0 else series * 0 + 0.5


def validate_dataframe(df: pd.DataFrame, required_cols: list[str], name: str = "DataFrame") -> None:
    """Raise ValueError if any required columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}. Got: {list(df.columns)}")


def annualised_return(monthly_rate: float) -> float:
    """Convert a monthly rate to an annualised rate."""
    return (1 + monthly_rate) ** 12 - 1


def monthly_rate(annual_rate: float) -> float:
    """Convert an annual rate to a monthly rate."""
    return (1 + annual_rate) ** (1 / 12) - 1


def format_currency(value: float, symbol: str = "$") -> str:
    """Format a float as a currency string with commas."""
    return f"{symbol}{value:,.2f}"


def country_kpi_summary(
    stations:  pd.DataFrame,
    customers: pd.DataFrame,
    revenue:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a one-row-per-country KPI summary table.

    Parameters
    ----------
    stations : pd.DataFrame
        Must have columns: country, status.
    customers : pd.DataFrame
        Must have columns: country.
    revenue : pd.DataFrame
        Must have columns: country, month, revenue_usd.

    Returns
    -------
    pd.DataFrame
        Columns: country, active_stations, total_stations,
        customers, latest_monthly_rev.
    """
    rows = []
    for country in stations["country"].unique():
        s   = stations[stations["country"] == country]
        c   = customers[customers["country"] == country]
        r   = revenue[revenue["country"] == country]
        rev = r[r["month"] == r["month"].max()]["revenue_usd"].sum() if len(r) > 0 else 0
        rows.append({
            "country":          country,
            "active_stations":  int((s["status"] == "active").sum()),
            "total_stations":   len(s),
            "customers":        len(c),
            "latest_monthly_rev": round(float(rev), 0),
        })
    return pd.DataFrame(rows).sort_values("latest_monthly_rev", ascending=False)


AFRICAN_EV_MARKETS = {
    "Kenya":   {"cities": ["Nairobi", "Mombasa", "Kisumu", "Nakuru"], "currency": "KES", "fx_to_usd": 0.0077},
    "Nigeria": {"cities": ["Lagos", "Abuja", "Kano", "Port Harcourt"], "currency": "NGN", "fx_to_usd": 0.00063},
    "Rwanda":  {"cities": ["Kigali", "Butare", "Gisenyi"],              "currency": "RWF", "fx_to_usd": 0.00071},
    "Uganda":  {"cities": ["Kampala", "Entebbe", "Jinja"],              "currency": "UGX", "fx_to_usd": 0.00027},
    "Ghana":   {"cities": ["Accra", "Kumasi", "Tamale"],                "currency": "GHS", "fx_to_usd": 0.066},
    "Ethiopia":{"cities": ["Addis Ababa", "Dire Dawa"],                 "currency": "ETB", "fx_to_usd": 0.0088},
}
