"""
afrikana.forecast
=================
Time-series demand forecasting for African mobility networks.

Uses Holt-Winters Exponential Smoothing with additive trend and
optional weekly seasonality. Falls back to a naive mean if the
series is too short for seasonal decomposition.

Example
-------
>>> from afrikana.forecast import DemandForecaster
>>>
>>> forecaster = DemandForecaster()
>>> daily      = forecaster.prepare_daily(swap_events_df)
>>> forecast   = forecaster.predict(daily, periods=30)
>>> print(forecast.head())
>>> peaks = forecaster.peak_hours(swap_events_df)
"""

from __future__ import annotations

import warnings
import pandas as pd
from dataclasses import dataclass
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")


@dataclass
class ForecastConfig:
    """Configuration for DemandForecaster."""

    seasonal_periods: int = 7
    confidence_width: float = 1.5
    min_series_length: int = 14


class DemandForecaster:
    """
    Daily/monthly demand forecaster using Holt-Winters Exponential Smoothing.

    Designed for swap event data but works with any time-indexed count series.

    Parameters
    ----------
    seasonal_periods : int
        Number of periods in a seasonal cycle. Default 7 (weekly for daily data).
    confidence_width : float
        Width of confidence interval in residual standard deviations. Default 1.5.

    Example
    -------
    >>> fc = DemandForecaster()
    >>> daily    = fc.prepare_daily(swap_events_df)
    >>> forecast = fc.predict(daily, periods=14)
    >>> print(f"Avg forecast: {forecast['forecast'].mean():.1f} swaps/day")
    """

    def __init__(
        self,
        seasonal_periods: int = 7,
        confidence_width: float = 1.5,
    ):
        self.config = ForecastConfig(
            seasonal_periods=seasonal_periods,
            confidence_width=confidence_width,
        )

    def prepare_daily(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        station_id: str | None = None,
    ) -> pd.Series:
        """
        Aggregate swap events to a daily count series.

        Parameters
        ----------
        df : pd.DataFrame
            Swap events DataFrame with a date column.
        date_col : str
            Column containing the date. Default "date".
        station_id : str, optional
            If provided, filter to this station only.

        Returns
        -------
        pd.Series
            Daily swap counts indexed by date, with zero-filled gaps.
        """
        data = df.copy()
        if station_id:
            data = data[data["station_id"] == station_id]
        data[date_col] = pd.to_datetime(data[date_col])
        series = data.groupby(date_col).size().rename("swaps")
        return series.asfreq("D", fill_value=0)

    def prepare_monthly(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        country: str | None = None,
        station_ids: list | None = None,
    ) -> pd.Series:
        """
        Aggregate swap events to a monthly count series.

        Parameters
        ----------
        df : pd.DataFrame
            Swap events DataFrame.
        date_col : str
            Column containing the date.
        country : str, optional
            Filter by country (requires station lookup — pass station_ids instead).
        station_ids : list, optional
            Filter to these station IDs.

        Returns
        -------
        pd.Series
            Monthly swap counts indexed by month start date.
        """
        data = df.copy()
        if station_ids:
            data = data[data["station_id"].isin(station_ids)]
        data[date_col] = pd.to_datetime(data[date_col])
        data["month"] = data[date_col].dt.to_period("M")
        series = data.groupby("month").size().rename("swaps")
        series.index = series.index.to_timestamp()
        return series

    def predict(self, series: pd.Series, periods: int = 30) -> pd.DataFrame:
        """
        Forecast demand for the next N periods.

        Parameters
        ----------
        series : pd.Series
            Time-indexed count series (output of prepare_daily or prepare_monthly).
        periods : int
            Number of periods to forecast ahead. Default 30.

        Returns
        -------
        pd.DataFrame
            Columns: date, forecast, lower, upper.
        """
        cfg = self.config
        freq = pd.infer_freq(series.index) or "D"

        try:
            use_seasonal = len(series) >= cfg.min_series_length
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add" if use_seasonal else None,
                seasonal_periods=cfg.seasonal_periods if use_seasonal else None,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            forecast = fit.forecast(periods)
            std = fit.resid.std()
        except Exception:
            last_val = float(series.iloc[-7:].mean()) if len(series) >= 7 else float(series.mean())
            forecast = pd.Series([last_val] * periods)
            std = last_val * 0.1

        step = pd.tseries.frequencies.to_offset(freq)
        future_idx = pd.date_range(
            start=series.index[-1] + step,
            periods=periods,
            freq=freq,
        )

        return pd.DataFrame(
            {
                "date": future_idx,
                "forecast": forecast.values.clip(0),
                "lower": (forecast.values - cfg.confidence_width * std).clip(0),
                "upper": (forecast.values + cfg.confidence_width * std).clip(0),
            }
        )

    def peak_hours(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Analyse peak demand by hour of day.

        Parameters
        ----------
        df : pd.DataFrame
            Swap events DataFrame with a timestamp column.
        timestamp_col : str
            Column containing the timestamp. Default "timestamp".

        Returns
        -------
        pd.DataFrame
            Columns: hour (0-23), swaps (count), pct (% of daily total).
        """
        data = df.copy()
        data["hour"] = pd.to_datetime(data[timestamp_col]).dt.hour
        counts = data.groupby("hour").size()
        hourly = counts.reindex(range(24), fill_value=0).reset_index()
        hourly.columns = ["hour", "swaps"]
        total = hourly["swaps"].sum()
        hourly["pct"] = (hourly["swaps"] / total * 100).round(1) if total > 0 else 0.0
        return hourly

    def forecast_summary(self, forecast_df: pd.DataFrame) -> dict:
        """
        Summary statistics for a forecast DataFrame.

        Returns
        -------
        dict
            Keys: periods, avg_forecast, peak_forecast, total_forecast,
            peak_date.
        """
        peak_idx = forecast_df["forecast"].idxmax()
        return {
            "periods": len(forecast_df),
            "avg_forecast": round(float(forecast_df["forecast"].mean()), 1),
            "peak_forecast": round(float(forecast_df["forecast"].max()), 1),
            "total_forecast": round(float(forecast_df["forecast"].sum()), 0),
            "peak_date": str(forecast_df.loc[peak_idx, "date"].date()),
        }

    def __repr__(self) -> str:
        return (
            f"DemandForecaster("
            f"seasonal_periods={self.config.seasonal_periods}, "
            f"confidence_width={self.config.confidence_width})"
        )
