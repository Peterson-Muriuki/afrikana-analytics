# afrikana-analytics

[![CI](https://github.com/Peterson-Muriuki/afrikana-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Peterson-Muriuki/afrikana-analytics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Packages](https://img.shields.io/badge/GitHub-Packages-181717?logo=github)](https://github.com/Peterson-Muriuki/afrikana-analytics/packages)

**Reusable analytics toolkit for African mobility, fintech, and EV swap station networks.**

Built from real analytical work on EV battery swap operations across Kenya, Nigeria, Rwanda, and Uganda. Covers the full data-to-decision stack: churn prediction, lifetime value, financial modelling, station deployment optimisation, and demand forecasting.

---

## Installation

```bash
# From GitHub Packages
pip install afrikana-analytics

# From source
git clone https://github.com/Peterson-Muriuki/afrikana-analytics
cd afrikana-analytics
pip install -e .
```

---

## Quick Start

```python
from afrikana.churn    import ChurnScorer
from afrikana.ltv      import LTVCalculator
from afrikana.financial import FinancialModel
from afrikana.stations  import StationOptimizer
from afrikana.forecast  import DemandForecaster

# --- Churn prediction ---
scorer = ChurnScorer()
scorer.fit(customers_df)
at_risk = scorer.at_risk(customers_df, threshold=0.5)
print(f"At-risk customers: {len(at_risk)}")
print(scorer.feature_importances())

# --- Customer LTV ---
calc   = LTVCalculator(gross_margin=0.62)
result = calc.compute(customers_df)
print(calc.tier_summary(result))
print(calc.revenue_at_risk(result))

# --- Station financial model ---
model = FinancialModel(swap_price_usd=2.50, n_stations=20)
print(model.unit_economics())
print(model.breakeven())
print(model.dcf())
print(model.scenarios())
mc = model.monte_carlo(n_sims=2000)
print(f"NPV P50: ${mc['npv_p50']:,.0f}  Prob +ve NPV: {mc['prob_positive_npv']}%")

# --- Deployment optimisation ---
opt        = StationOptimizer()
candidates = opt.generate_grid((-1.286389, 36.817223), n=40)
scored     = opt.score(candidates, existing_stations_df)
print(opt.recommend(scored, top_n=5))

# --- Demand forecasting ---
fc       = DemandForecaster()
daily    = fc.prepare_daily(swap_events_df)
forecast = fc.predict(daily, periods=30)
print(fc.forecast_summary(forecast))
```

---

## Modules

### `ChurnScorer`
Gradient Boosting churn predictor for subscription/usage-based mobility businesses.

```python
from afrikana.churn import ChurnScorer, ChurnScorerConfig

config = ChurnScorerConfig(n_estimators=200, verbose=True)
scorer = ChurnScorer(config)
scorer.fit(df)                       # trains and evaluates on held-out split
scored = scorer.score(df)           # adds churn_score [0-1] and churn_risk tier
at_risk = scorer.at_risk(df, 0.6)  # customers above threshold, sorted
scorer.feature_importances()        # what drives churn most
print(scorer.summary())             # {"auc": 0.82, "top_feature": "last_swap_days_ago", ...}
```

**Required columns:** `swap_freq_monthly`, `last_swap_days_ago`, `tenure_months`, `monthly_revenue`, `churned` (0/1 target)

---

### `LTVCalculator`
Discounted survival-adjusted Customer Lifetime Value with Bronze/Silver/Gold tiers.

```python
from afrikana.ltv import LTVCalculator

calc = LTVCalculator(gross_margin=0.62, discount_rate_annual=0.12, max_horizon_months=36)
df   = calc.compute(customers_df)      # adds ltv, ltv_tier, expected_lifetime
calc.tier_summary(df)                  # count, avg_ltv, total_ltv per tier
calc.segment_summary(df, "country")    # LTV by country / segment / city
calc.revenue_at_risk(df, threshold=0.5)
```

**Required columns:** `churn_probability` (or `churn_score`), `monthly_revenue`, `tenure_months`

---

### `FinancialModel`
Full financial model for an EV swap station network.

```python
from afrikana.financial import FinancialModel

model = FinancialModel(
    swap_price_usd=2.50,
    swaps_per_station_day=18,
    n_stations=20,
    gross_margin_pct=0.62,
    discount_rate_annual=0.12,
)

model.unit_economics()   # per-station P&L: revenue, EBITDA, NOPAT, contribution/swap
model.pl_projection()    # 36-month network P&L as DataFrame
model.cash_flow()        # operating CF, capex, FCF, cumulative cash
model.breakeven()        # swaps/day needed, margin of safety, payback period
model.dcf()              # NPV, IRR, ROI, terminal value
model.scenarios()        # Base / Bull / Bear comparison table
model.monte_carlo(2000)  # P10/P50/P90 NPV distribution, VaR 95%, prob +ve NPV
```

---

### `StationOptimizer`
Multi-criteria deployment scorer for new swap station locations.

```python
from afrikana.stations import StationOptimizer, OptimizerConfig

config = OptimizerConfig(
    weight_demand_density=0.30,
    weight_coverage_gap=0.25,
    weight_revenue_potential=0.25,
    weight_underserved=0.20,
)
opt = StationOptimizer(config)

candidates  = opt.generate_grid((-1.286, 36.817), n=40)  # synthetic grid
scored      = opt.score(candidates, existing_stations_df)
top5        = opt.recommend(scored, top_n=5)
stats       = opt.coverage_stats(scored)
```

**Required columns in existing_stations_df:** `lat`, `lon`, `status`

---

### `DemandForecaster`
Holt-Winters time-series forecaster with confidence intervals.

```python
from afrikana.forecast import DemandForecaster

fc       = DemandForecaster(seasonal_periods=7)
daily    = fc.prepare_daily(swap_events_df)
monthly  = fc.prepare_monthly(swap_events_df)
forecast = fc.predict(daily, periods=30)   # date, forecast, lower, upper
summary  = fc.forecast_summary(forecast)
peaks    = fc.peak_hours(swap_events_df)   # 24-row hour-of-day breakdown
```

---

## Running the Demo

```bash
pip install -e .
python examples/spiro_demo.py
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=afrikana
```

---

## Target Markets

Built for and tested against data patterns from:

| Country  | Cities                              | Currency |
|----------|-------------------------------------|----------|
| Kenya    | Nairobi, Mombasa, Kisumu, Nakuru    | KES      |
| Nigeria  | Lagos, Abuja, Kano, Port Harcourt   | NGN      |
| Rwanda   | Kigali, Butare, Gisenyi             | RWF      |
| Uganda   | Kampala, Entebbe, Jinja             | UGX      |
| Ghana    | Accra, Kumasi, Tamale               | GHS      |
| Ethiopia | Addis Ababa, Dire Dawa              | ETB      |

---

## Author

**Peterson Mutegi** — Data Analyst · AI Engineer · Financial Engineer  
Nairobi, Kenya · [pitmuriuki@gmail.com](mailto:pitmuriuki@gmail.com)  
[GitHub](https://github.com/Peterson-Muriuki) · [LinkedIn]https://www.linkedin.com/in/peterson-muriuki-5857aaa9/)

Built on top of real analytical work for African EV mobility operations.

---

## License

MIT — see [LICENSE](LICENSE) for details.
