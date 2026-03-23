"""tests/test_ltv.py"""
import pytest
from afrikana.ltv import LTVCalculator


def test_compute_adds_ltv_column(customers_df):
    calc   = LTVCalculator()
    result = calc.compute(customers_df)
    assert "ltv" in result.columns


def test_ltv_non_negative(customers_df):
    calc   = LTVCalculator()
    result = calc.compute(customers_df)
    assert (result["ltv"] >= 0).all()


def test_ltv_tier_labels(customers_df):
    calc   = LTVCalculator()
    result = calc.compute(customers_df)
    valid  = {"Bronze", "Silver", "Gold"}
    assert set(result["ltv_tier"].dropna().unique()).issubset(valid)


def test_tier_summary_has_three_rows(customers_df):
    calc    = LTVCalculator()
    result  = calc.compute(customers_df)
    summary = calc.tier_summary(result)
    assert len(summary) == 3


def test_revenue_at_risk_keys(customers_df):
    from afrikana.churn import ChurnScorer
    scored = ChurnScorer().fit(customers_df).score(customers_df)
    calc   = LTVCalculator()
    result = calc.compute(scored)
    rar    = calc.revenue_at_risk(result)
    for key in ["n_at_risk", "monthly_revenue_at_risk", "ltv_at_risk", "pct_of_total_ltv"]:
        assert key in rar


def test_higher_margin_means_higher_ltv(customers_df):
    calc_lo = LTVCalculator(gross_margin=0.40)
    calc_hi = LTVCalculator(gross_margin=0.80)
    lo = calc_lo.compute(customers_df)["ltv"].mean()
    hi = calc_hi.compute(customers_df)["ltv"].mean()
    assert hi > lo


def test_higher_churn_means_lower_ltv(customers_df):
    import pandas as pd, numpy as np
    low_churn = customers_df.copy()
    low_churn["churn_probability"] = 0.05
    high_churn = customers_df.copy()
    high_churn["churn_probability"] = 0.80
    calc = LTVCalculator()
    assert calc.compute(low_churn)["ltv"].mean() > calc.compute(high_churn)["ltv"].mean()
