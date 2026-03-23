"""
examples/spiro_demo.py
======================
End-to-end demonstration of the afrikana-analytics package
using a simulated Spiro swap station dataset.

Run:
    pip install afrikana-analytics
    python examples/spiro_demo.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── 1. Generate demo data ──────────────────────────────────────────────────
print("=" * 60)
print("afrikana-analytics — Spiro Demo")
print("=" * 60)

rng = np.random.default_rng(42)
n_customers = 500
n_stations  = 40
n_events    = 8000

customers = pd.DataFrame({
    "customer_id":       [f"C{i:05d}" for i in range(n_customers)],
    "country":           rng.choice(["Kenya", "Nigeria", "Rwanda", "Uganda"], n_customers),
    "segment":           rng.choice(["commuter", "delivery", "logistics", "casual"], n_customers),
    "swap_freq_monthly": np.maximum(0, rng.normal(18, 7, n_customers)).round().astype(int),
    "last_swap_days_ago":rng.exponential(12, n_customers).astype(int),
    "tenure_months":     rng.uniform(1, 36, n_customers).round(1),
    "monthly_revenue":   np.maximum(5, rng.normal(45, 18, n_customers)).round(2),
    "churn_probability": rng.uniform(0, 1, n_customers).round(4),
    "churned":           rng.choice([0, 1], n_customers, p=[0.85, 0.15]),
})

stations = pd.DataFrame({
    "station_id": [f"ST{i:04d}" for i in range(n_stations)],
    "country":    rng.choice(["Kenya", "Nigeria", "Rwanda", "Uganda"], n_stations),
    "lat":        rng.uniform(-5, 12, n_stations),
    "lon":        rng.uniform(29, 40, n_stations),
    "status":     rng.choice(["active", "inactive"], n_stations, p=[0.85, 0.15]),
})

base = datetime.today() - timedelta(days=365)
events = pd.DataFrame({
    "station_id": [f"ST{i:04d}" for i in rng.integers(0, n_stations, n_events)],
    "timestamp":  [base + timedelta(days=int(d), hours=int(h))
                   for d, h in zip(rng.integers(0, 365, n_events), rng.integers(5, 23, n_events))],
})
events["date"] = pd.to_datetime(events["timestamp"]).dt.date
events["hour"] = pd.to_datetime(events["timestamp"]).dt.hour

# ── 2. Churn Scoring ───────────────────────────────────────────────────────
print("\n── ChurnScorer ──────────────────────────────────────────")
from afrikana.churn import ChurnScorer

scorer   = ChurnScorer()
scorer.fit(customers)
scored   = scorer.score(customers)
at_risk  = scorer.at_risk(customers, threshold=0.5)

print(f"Model AUC          : {scorer.auc_:.3f}")
print(f"Customers scored   : {len(scored)}")
print(f"At-risk (≥0.5)     : {len(at_risk)}")
print(f"Top churn driver   : {scorer.feature_importances().iloc[0]['feature']}")
print(scorer)

# ── 3. LTV Calculation ────────────────────────────────────────────────────
print("\n── LTVCalculator ────────────────────────────────────────")
from afrikana.ltv import LTVCalculator

calc  = LTVCalculator(gross_margin=0.62, discount_rate_annual=0.12)
ltv_df = calc.compute(scored)

print(f"Avg LTV            : ${ltv_df['ltv'].mean():,.2f}")
print(f"Total portfolio LTV: ${ltv_df['ltv'].sum():,.0f}")
print("\nTier breakdown:")
print(calc.tier_summary(ltv_df).to_string(index=False))

rar = calc.revenue_at_risk(ltv_df)
print(f"\nRevenue at risk    : ${rar['monthly_revenue_at_risk']:,.2f}/month")
print(f"LTV at risk        : ${rar['ltv_at_risk']:,.0f} ({rar['pct_of_total_ltv']}%)")

# ── 4. Financial Model ────────────────────────────────────────────────────
print("\n── FinancialModel ───────────────────────────────────────")
from afrikana.financial import FinancialModel

model = FinancialModel(swap_price_usd=2.50, n_stations=20, swaps_per_station_day=18)
ue    = model.unit_economics()
be    = model.breakeven()
dcf   = model.dcf()

print(f"Revenue/station/month : ${ue['gross_revenue']:,.2f}")
print(f"EBITDA/station/month  : ${ue['ebitda']:,.2f} ({ue['ebitda_margin_pct']:.1f}%)")
print(f"Contribution/swap     : ${ue['contribution_per_swap']:.4f}")
print(f"Breakeven swaps/day   : {be['breakeven_swaps_per_day']}")
print(f"Margin of safety      : {be['margin_of_safety_pct']:.1f}%")
print(f"Payback               : {be['payback_months']} months")
print(f"NPV (36M)             : ${dcf['npv']:,.0f}")

print("\nScenario comparison:")
print(model.scenarios()[["Scenario","Revenue/Stn ($)","EBITDA/Stn ($)","NPV ($)","Payback (months)"]].to_string(index=False))

mc = model.monte_carlo(n_sims=500)
print(f"\nMonte Carlo (500 sims):")
print(f"  NPV P10/P50/P90   : ${mc['npv_p10']:,.0f} / ${mc['npv_p50']:,.0f} / ${mc['npv_p90']:,.0f}")
print(f"  Prob +ve NPV      : {mc['prob_positive_npv']}%")
print(f"  VaR 95%           : ${mc['var_95']:,.0f}")

# ── 5. Station Optimizer ──────────────────────────────────────────────────
print("\n── StationOptimizer ─────────────────────────────────────")
from afrikana.stations import StationOptimizer

opt        = StationOptimizer()
candidates = opt.generate_grid((-1.286389, 36.817223), n=40)
scored_locs = opt.score(candidates, stations)
top5       = opt.recommend(scored_locs, top_n=5)
stats      = opt.coverage_stats(scored_locs)

print(f"Candidates scored  : {stats['total_candidates']}")
print(f"High priority      : {stats['high_priority']}")
print(f"Avg coverage gap   : {stats['avg_coverage_gap_km']:.2f} km")
print("\nTop 5 deployment sites:")
print(top5[["candidate_id","deployment_score","priority","nearest_station_km"]].to_string(index=False))

# ── 6. Demand Forecast ────────────────────────────────────────────────────
print("\n── DemandForecaster ─────────────────────────────────────")
from afrikana.forecast import DemandForecaster

fc       = DemandForecaster()
daily    = fc.prepare_daily(events)
forecast = fc.predict(daily, periods=14)
summary  = fc.forecast_summary(forecast)
peaks    = fc.peak_hours(events)

print(f"Historical days    : {len(daily)}")
print(f"Avg forecast       : {summary['avg_forecast']} swaps/day")
print(f"Peak forecast      : {summary['peak_forecast']} swaps on {summary['peak_date']}")
print(f"Peak hour          : {peaks.loc[peaks['swaps'].idxmax(), 'hour']}:00 "
      f"({peaks['swaps'].max()} swaps, {peaks.loc[peaks['swaps'].idxmax(), 'pct']:.1f}%)")

print("\n" + "=" * 60)
print("Demo complete.")
print("=" * 60)
