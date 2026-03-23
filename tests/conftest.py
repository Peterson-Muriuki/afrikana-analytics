"""
tests/conftest.py
Shared pytest fixtures for the afrikana-analytics test suite.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)


@pytest.fixture
def customers_df():
    """Synthetic customer DataFrame matching expected schema."""
    n = 300
    rng = np.random.default_rng(42)
    swap_freq = np.maximum(0, rng.normal(18, 7, n)).round().astype(int)
    last_swap = rng.exponential(12, n).astype(int)
    tenure = rng.uniform(1, 36, n).round(1)
    revenue = np.maximum(5, rng.normal(45, 18, n)).round(2)
    churn_p = np.clip(
        0.05 + (1 / (swap_freq + 1)) * 0.4 + (last_swap / 60) * 0.3 - (tenure / 36) * 0.1, 0, 1
    )
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "country": rng.choice(["Kenya", "Nigeria", "Rwanda", "Uganda"], n),
            "city": rng.choice(["Nairobi", "Lagos", "Kigali", "Kampala"], n),
            "segment": rng.choice(["commuter", "delivery", "logistics", "casual"], n),
            "swap_freq_monthly": swap_freq,
            "last_swap_days_ago": last_swap,
            "tenure_months": tenure,
            "monthly_revenue": revenue,
            "churn_probability": churn_p.round(4),
            "churned": (rng.random(n) < churn_p).astype(int),
        }
    )


@pytest.fixture
def stations_df():
    """Synthetic stations DataFrame."""
    n = 50
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "station_id": [f"ST{i:04d}" for i in range(n)],
            "country": rng.choice(["Kenya", "Nigeria", "Rwanda", "Uganda"], n),
            "city": rng.choice(["Nairobi", "Lagos", "Kigali", "Kampala"], n),
            "lat": rng.uniform(-5, 12, n),
            "lon": rng.uniform(29, 40, n),
            "capacity": rng.choice([10, 20, 30], n),
            "status": rng.choice(["active", "inactive", "maintenance"], n, p=[0.8, 0.1, 0.1]),
        }
    )


@pytest.fixture
def swap_events_df():
    """Synthetic swap events DataFrame."""
    n = 5000
    rng = np.random.default_rng(42)
    base = datetime.today() - timedelta(days=365)
    timestamps = [
        base + timedelta(days=int(d), hours=int(h))
        for d, h in zip(rng.integers(0, 365, n), rng.integers(5, 23, n))
    ]
    df = pd.DataFrame(
        {
            "swap_id": [f"SW{i:06d}" for i in range(n)],
            "station_id": [f"ST{i:04d}" for i in rng.integers(0, 50, n)],
            "customer_id": [f"C{i:05d}" for i in rng.integers(0, 300, n)],
            "timestamp": timestamps,
            "swap_successful": rng.choice([1, 0], n, p=[0.95, 0.05]),
        }
    )
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.strftime("%A")
    return df


@pytest.fixture
def financial_model():
    """Default FinancialModel instance."""
    from afrikana.financial import FinancialModel

    return FinancialModel()
