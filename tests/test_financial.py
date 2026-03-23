"""tests/test_financial.py"""
import pytest
import pandas as pd
from afrikana.financial import FinancialModel


def test_unit_economics_keys(financial_model):
    ue = financial_model.unit_economics()
    for key in ["gross_revenue", "ebitda", "ebitda_margin_pct", "contribution_per_swap"]:
        assert key in ue


def test_revenue_positive(financial_model):
    assert financial_model.unit_economics()["gross_revenue"] > 0


def test_pl_projection_length(financial_model):
    pl = financial_model.pl_projection()
    assert len(pl) == financial_model.config.projection_months


def test_pl_revenue_grows_with_stations(financial_model):
    pl = financial_model.pl_projection()
    assert pl["revenue"].iloc[-1] > pl["revenue"].iloc[0]


def test_cash_flow_columns(financial_model):
    cf = financial_model.cash_flow()
    for col in ["month", "operating_cf", "free_cash_flow", "cumulative_cash"]:
        assert col in cf.columns


def test_breakeven_swaps_positive(financial_model):
    be = financial_model.breakeven()
    assert be["breakeven_swaps_per_day"] > 0


def test_breakeven_margin_of_safety_type(financial_model):
    be = financial_model.breakeven()
    assert isinstance(be["margin_of_safety_pct"], float)


def test_dcf_keys(financial_model):
    dcf = financial_model.dcf()
    for key in ["initial_investment", "npv", "roi_pct"]:
        assert key in dcf


def test_higher_price_gives_higher_npv():
    m_lo = FinancialModel(swap_price_usd=1.50)
    m_hi = FinancialModel(swap_price_usd=4.00)
    assert m_hi.dcf()["npv"] > m_lo.dcf()["npv"]


def test_scenarios_has_three_rows(financial_model):
    sc = financial_model.scenarios()
    assert len(sc) == 3
    assert set(sc["Scenario"]) == {"Base", "Bull", "Bear"}


def test_bull_net_income_beats_bear(financial_model):
    """Bull scenario should generate more net income than Bear."""
    sc = financial_model.scenarios().set_index("Scenario")
    assert sc.loc["Bull", "36M Net Income ($)"] > sc.loc["Bear", "36M Net Income ($)"]


def test_bull_ebitda_beats_bear(financial_model):
    """Bull EBITDA margin should exceed Bear EBITDA margin."""
    sc = financial_model.scenarios().set_index("Scenario")
    assert sc.loc["Bull", "EBITDA Margin (%)"] > sc.loc["Bear", "EBITDA Margin (%)"]


def test_monte_carlo_keys(financial_model):
    mc = financial_model.monte_carlo(n_sims=100)
    for key in ["npv_p10", "npv_p50", "npv_p90", "prob_positive_npv", "var_95"]:
        assert key in mc


def test_monte_carlo_p10_le_p50_le_p90(financial_model):
    mc = financial_model.monte_carlo(n_sims=100)
    assert mc["npv_p10"] <= mc["npv_p50"] <= mc["npv_p90"]


def test_invalid_param_raises():
    with pytest.raises(ValueError, match="Unknown FinancialModel parameters"):
        FinancialModel(nonexistent_param=999)


def test_repr(financial_model):
    assert "FinancialModel" in repr(financial_model)
