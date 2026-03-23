"""tests/test_stations_forecast.py"""
import pytest
import pandas as pd
from afrikana.stations import StationOptimizer, OptimizerConfig
from afrikana.forecast import DemandForecaster


# ── StationOptimizer ──────────────────────────────────────────────────────────

def test_generate_grid_shape():
    opt  = StationOptimizer()
    grid = opt.generate_grid((-1.286, 36.817), n=30)
    assert len(grid) == 30
    assert "lat" in grid.columns and "lon" in grid.columns


def test_score_adds_deployment_score(stations_df):
    opt   = StationOptimizer()
    grid  = opt.generate_grid((-1.286, 36.817), n=20)
    scored = opt.score(grid, stations_df)
    assert "deployment_score" in scored.columns
    assert "priority" in scored.columns


def test_score_sorted_descending(stations_df):
    opt    = StationOptimizer()
    grid   = opt.generate_grid((-1.286, 36.817), n=20)
    scored = opt.score(grid, stations_df)
    scores = scored["deployment_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_recommend_returns_top_n(stations_df):
    opt    = StationOptimizer()
    grid   = opt.generate_grid((-1.286, 36.817), n=30)
    scored = opt.score(grid, stations_df)
    top5   = opt.recommend(scored, top_n=5)
    assert len(top5) == 5


def test_coverage_stats_keys(stations_df):
    opt    = StationOptimizer()
    grid   = opt.generate_grid((-1.286, 36.817), n=20)
    scored = opt.score(grid, stations_df)
    stats  = opt.coverage_stats(scored)
    for key in ["total_candidates", "high_priority", "avg_coverage_gap_km"]:
        assert key in stats


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        OptimizerConfig(
            weight_demand_density=0.5,
            weight_coverage_gap=0.5,
            weight_revenue_potential=0.5,
            weight_underserved=0.5,
        )


# ── DemandForecaster ──────────────────────────────────────────────────────────

def test_prepare_daily_returns_series(swap_events_df):
    fc     = DemandForecaster()
    series = fc.prepare_daily(swap_events_df)
    assert hasattr(series, "index")
    assert len(series) > 0


def test_prepare_daily_no_gaps(swap_events_df):
    fc     = DemandForecaster()
    series = fc.prepare_daily(swap_events_df)
    assert series.index.freq is not None or len(series) > 1


def test_predict_length(swap_events_df):
    fc       = DemandForecaster()
    series   = fc.prepare_daily(swap_events_df)
    forecast = fc.predict(series, periods=14)
    assert len(forecast) == 14


def test_predict_columns(swap_events_df):
    fc       = DemandForecaster()
    series   = fc.prepare_daily(swap_events_df)
    forecast = fc.predict(series, periods=7)
    assert set(["date", "forecast", "lower", "upper"]).issubset(forecast.columns)


def test_forecast_non_negative(swap_events_df):
    fc       = DemandForecaster()
    series   = fc.prepare_daily(swap_events_df)
    forecast = fc.predict(series, periods=14)
    assert (forecast["forecast"] >= 0).all()
    assert (forecast["lower"]   >= 0).all()


def test_lower_le_forecast_le_upper(swap_events_df):
    fc       = DemandForecaster()
    series   = fc.prepare_daily(swap_events_df)
    forecast = fc.predict(series, periods=14)
    assert (forecast["lower"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["upper"]).all()


def test_peak_hours_has_24_rows(swap_events_df):
    fc    = DemandForecaster()
    peaks = fc.peak_hours(swap_events_df)
    assert len(peaks) == 24


def test_peak_hours_pct_sums_to_100(swap_events_df):
    fc    = DemandForecaster()
    peaks = fc.peak_hours(swap_events_df)
    assert abs(peaks["pct"].sum() - 100.0) < 0.5


def test_forecast_summary_keys(swap_events_df):
    fc       = DemandForecaster()
    series   = fc.prepare_daily(swap_events_df)
    forecast = fc.predict(series, periods=14)
    summary  = fc.forecast_summary(forecast)
    for key in ["periods", "avg_forecast", "peak_forecast", "total_forecast", "peak_date"]:
        assert key in summary
