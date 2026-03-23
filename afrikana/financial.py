"""
afrikana.financial
==================
Financial modelling for African EV swap station networks.

Covers unit economics, P&L projection, cash flow, breakeven,
DCF valuation, Monte Carlo simulation, and scenario analysis.

Example
-------
>>> from afrikana.financial import FinancialModel
>>>
>>> model = FinancialModel(swap_price=2.50, n_stations=20, swaps_per_station_day=18)
>>> print(model.unit_economics())
>>> print(model.breakeven())
>>> print(model.dcf())
>>> scenarios = model.scenarios()
>>> mc = model.monte_carlo(n_sims=1000)
>>> print(f"NPV P50: ${mc['npv_p50']:,.0f}  Prob +ve: {mc['prob_positive_npv']}%")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")


@dataclass
class FinancialConfig:
    """All financial model assumption parameters."""

    swap_price_usd: float = 2.50
    swaps_per_station_day: float = 18.0
    active_days_per_month: int = 28
    station_capex_usd: float = 5000.0
    battery_cost_usd: float = 800.0
    batteries_per_station: int = 20
    capex_lifetime_years: int = 5
    opex_per_station_month: float = 220.0
    staff_cost_per_station: float = 80.0
    logistics_cost_pct: float = 0.08
    n_stations: int = 20
    station_growth_monthly: float = 0.03
    gross_margin_pct: float = 0.62
    discount_rate_annual: float = 0.12
    tax_rate: float = 0.25
    projection_months: int = 36


SCENARIO_OVERRIDES = {
    "Base": {},
    "Bull": {
        "swap_price_usd": 3.20,
        "swaps_per_station_day": 25.0,
        "opex_per_station_month": 200.0,
        "station_growth_monthly": 0.06,
    },
    "Bear": {
        "swap_price_usd": 1.80,
        "swaps_per_station_day": 10.0,
        "opex_per_station_month": 260.0,
        "station_growth_monthly": 0.01,
    },
}


class FinancialModel:
    """
    Financial model for an EV battery swap station network.

    All monetary values are in USD. All rates are expressed as decimals
    (e.g. 0.12 = 12%).

    Parameters
    ----------
    **kwargs
        Any field from FinancialConfig, passed as keyword arguments.
        E.g. FinancialModel(swap_price_usd=3.0, n_stations=50)

    Example
    -------
    >>> model = FinancialModel(swap_price_usd=2.50, n_stations=20)
    >>> ue = model.unit_economics()
    >>> print(f"EBITDA/station/month: ${ue['ebitda']:,.2f}")
    >>> print(f"EBITDA margin: {ue['ebitda_margin_pct']:.1f}%")
    """

    def __init__(self, **kwargs):
        cfg_fields = FinancialConfig.__dataclass_fields__.keys()
        valid = {k: v for k, v in kwargs.items() if k in cfg_fields}
        invalid = set(kwargs) - set(valid)
        if invalid:
            raise ValueError(f"Unknown FinancialModel parameters: {invalid}")
        self.config = FinancialConfig(**valid)

    # ── Unit economics ────────────────────────────────────────────────────────

    def unit_economics(self) -> dict:
        """
        Revenue and cost breakdown per station per month.

        Returns
        -------
        dict
            Keys: monthly_swaps, gross_revenue, capex_monthly_amortised,
            logistics_cost, total_opex, gross_profit, ebitda, ebitda_margin_pct,
            ebit, nopat, contribution_per_swap, contribution_margin_pct,
            revenue_per_battery_day.
        """
        p = self.config
        monthly_swaps = p.swaps_per_station_day * p.active_days_per_month
        gross_revenue = monthly_swaps * p.swap_price_usd
        capex_monthly = (p.station_capex_usd + p.batteries_per_station * p.battery_cost_usd) / (
            p.capex_lifetime_years * 12
        )
        logistics_cost = gross_revenue * p.logistics_cost_pct
        total_opex = p.opex_per_station_month + p.staff_cost_per_station + logistics_cost
        gross_profit = gross_revenue * p.gross_margin_pct
        ebitda = gross_profit - p.opex_per_station_month - p.staff_cost_per_station - logistics_cost
        ebit = ebitda - capex_monthly
        nopat = ebit * (1 - p.tax_rate)
        var_cost = (
            p.swap_price_usd * (1 - p.gross_margin_pct) + p.swap_price_usd * p.logistics_cost_pct
        )
        contrib = p.swap_price_usd - var_cost

        return {
            "monthly_swaps": round(monthly_swaps, 0),
            "gross_revenue": round(gross_revenue, 2),
            "capex_monthly_amortised": round(capex_monthly, 2),
            "logistics_cost": round(logistics_cost, 2),
            "total_opex": round(total_opex, 2),
            "gross_profit": round(gross_profit, 2),
            "ebitda": round(ebitda, 2),
            "ebitda_margin_pct": round(ebitda / gross_revenue * 100, 1) if gross_revenue else 0,
            "ebit": round(ebit, 2),
            "nopat": round(nopat, 2),
            "contribution_per_swap": round(contrib, 4),
            "contribution_margin_pct": round(contrib / p.swap_price_usd * 100, 1)
            if p.swap_price_usd
            else 0,
            "revenue_per_battery_day": round(
                gross_revenue / p.batteries_per_station / p.active_days_per_month, 3
            ),
        }

    # ── P&L projection ────────────────────────────────────────────────────────

    def pl_projection(self) -> pd.DataFrame:
        """
        Month-by-month P&L for the full station network over projection_months.

        Returns
        -------
        pd.DataFrame
            Columns: month, n_stations, revenue, cogs, gross_profit, opex,
            ebitda, depreciation, ebit, tax, net_income, ebitda_margin.
        """
        p = self.config
        ue = self.unit_economics()
        rows = []

        for m in range(1, p.projection_months + 1):
            n = p.n_stations * ((1 + p.station_growth_monthly) ** m)
            revenue = ue["gross_revenue"] * n
            cogs = revenue * (1 - p.gross_margin_pct)
            gross_profit = revenue - cogs
            opex = (p.opex_per_station_month + p.staff_cost_per_station) * n
            logistics = revenue * p.logistics_cost_pct
            ebitda = gross_profit - opex - logistics
            depreciation = ue["capex_monthly_amortised"] * n
            ebit = ebitda - depreciation
            tax = max(0, ebit * p.tax_rate)
            rows.append(
                {
                    "month": m,
                    "n_stations": round(n, 1),
                    "revenue": round(revenue, 0),
                    "cogs": round(cogs, 0),
                    "gross_profit": round(gross_profit, 0),
                    "opex": round(opex + logistics, 0),
                    "ebitda": round(ebitda, 0),
                    "depreciation": round(depreciation, 0),
                    "ebit": round(ebit, 0),
                    "tax": round(tax, 0),
                    "net_income": round(ebit - tax, 0),
                    "ebitda_margin": round(ebitda / revenue * 100, 1) if revenue else 0,
                }
            )
        return pd.DataFrame(rows)

    # ── Cash flow ─────────────────────────────────────────────────────────────

    def cash_flow(self) -> pd.DataFrame:
        """
        Operating and investing cash flows, and cumulative cash position.

        Returns
        -------
        pd.DataFrame
            Columns: month, operating_cf, capex, free_cash_flow, cumulative_cash.
        """
        p = self.config
        pl = self.pl_projection()
        spu = p.station_capex_usd + p.batteries_per_station * p.battery_cost_usd

        cum = -spu * p.n_stations
        rows = []

        for i, row in pl.iterrows():
            prev_n = pl.loc[i - 1, "n_stations"] if i > 0 else p.n_stations
            capex = max(0, row["n_stations"] - prev_n) * spu
            ocf = row["net_income"] + row["depreciation"]
            fcf = ocf - capex
            cum += fcf
            rows.append(
                {
                    "month": row["month"],
                    "operating_cf": round(ocf, 0),
                    "capex": round(-capex, 0),
                    "free_cash_flow": round(fcf, 0),
                    "cumulative_cash": round(cum, 0),
                }
            )
        return pd.DataFrame(rows)

    # ── Breakeven ─────────────────────────────────────────────────────────────

    def breakeven(self) -> dict:
        """
        Breakeven analysis — swaps per day needed to cover all costs.

        Returns
        -------
        dict
            Keys: breakeven_swaps_per_day, breakeven_swaps_per_month,
            contribution_per_swap, fixed_costs_monthly, payback_months,
            margin_of_safety_pct, current_swaps_month, variable_cost_per_swap.
        """
        p = self.config
        ue = self.unit_economics()

        capex_monthly = ue["capex_monthly_amortised"]
        fixed_costs = p.opex_per_station_month + p.staff_cost_per_station + capex_monthly
        contrib = ue["contribution_per_swap"]

        if contrib <= 0:
            be_month = float("inf")
        else:
            be_month = fixed_costs / contrib

        be_day = be_month / p.active_days_per_month

        cf = self.cash_flow()
        pos = cf[cf["cumulative_cash"] >= 0]
        payback = int(pos["month"].min()) if len(pos) > 0 else None

        current = p.swaps_per_station_day * p.active_days_per_month
        mos = ((current - be_month) / current * 100) if current else 0

        return {
            "breakeven_swaps_per_day": round(be_day, 1),
            "breakeven_swaps_per_month": round(be_month, 0),
            "contribution_per_swap": round(contrib, 4),
            "variable_cost_per_swap": round(ue["contribution_per_swap"] - contrib + contrib, 4),
            "fixed_costs_monthly": round(fixed_costs, 2),
            "payback_months": payback if payback else ">36",
            "margin_of_safety_pct": round(mos, 1),
            "current_swaps_month": round(current, 0),
        }

    # ── DCF valuation ─────────────────────────────────────────────────────────

    def dcf(self, terminal_growth_annual: float = 0.02) -> dict:
        """
        Discounted Cash Flow valuation of the station network.

        Parameters
        ----------
        terminal_growth_annual : float
            Long-run FCF growth rate for terminal value. Default 0.02 (2%).

        Returns
        -------
        dict
            Keys: initial_investment, npv, npv_with_terminal, terminal_value,
            irr_annual_pct, roi_pct, total_revenue_36m, total_net_income_36m.
        """
        p = self.config
        pl = self.pl_projection()
        cf = self.cash_flow()
        spu = p.station_capex_usd + p.batteries_per_station * p.battery_cost_usd

        r_m = (1 + p.discount_rate_annual) ** (1 / 12) - 1
        capex0 = spu * p.n_stations

        flows = [-capex0] + cf["free_cash_flow"].tolist()
        disc = [c / (1 + r_m) ** t for t, c in enumerate(flows)]
        npv = sum(disc)

        g_m = (1 + terminal_growth_annual) ** (1 / 12) - 1
        last_fcf = cf["free_cash_flow"].iloc[-1]
        if r_m > g_m and last_fcf > 0:
            tv = last_fcf * (1 + g_m) / (r_m - g_m)
            tv_pv = tv / (1 + r_m) ** len(flows)
        else:
            tv_pv = 0

        try:
            irr_m = self._irr(flows)
            irr_ann = (1 + irr_m) ** 12 - 1
            irr_ann = irr_ann if -1 < irr_ann < 10 else None
        except Exception:
            irr_ann = None

        roi = pl["net_income"].sum() / capex0 * 100 if capex0 else 0

        return {
            "initial_investment": round(capex0, 0),
            "npv": round(npv, 0),
            "npv_with_terminal": round(npv + tv_pv, 0),
            "terminal_value": round(tv_pv, 0),
            "irr_annual_pct": round(irr_ann * 100, 1) if irr_ann else None,
            "roi_pct": round(roi, 1),
            "total_revenue_36m": round(pl["revenue"].sum(), 0),
            "total_net_income_36m": round(pl["net_income"].sum(), 0),
        }

    # ── Monte Carlo ───────────────────────────────────────────────────────────

    def monte_carlo(self, n_sims: int = 2000, random_seed: int = 42) -> dict:
        """
        Monte Carlo simulation sampling key assumptions.

        Sampled variables: swap_price_usd (±15%), swaps_per_station_day (±20%),
        opex_per_station_month (±12%), discount_rate_annual (±2pp).

        Parameters
        ----------
        n_sims : int
            Number of simulations. Default 2000.
        random_seed : int
            NumPy random seed for reproducibility. Default 42.

        Returns
        -------
        dict
            Keys: npv_p10, npv_p50, npv_p90, npv_mean, prob_positive_npv,
            var_95, rev_p10, rev_p50, rev_p90, npv_simulations (array),
            revenue_simulations (array).
        """
        p = self.config
        rng = np.random.default_rng(random_seed)
        npvs, revs = [], []

        for _ in range(n_sims):
            try:
                sim = FinancialModel(
                    **{
                        **self._config_dict(),
                        "swap_price_usd": max(
                            0.5, rng.normal(p.swap_price_usd, p.swap_price_usd * 0.15)
                        ),
                        "swaps_per_station_day": max(
                            1.0, rng.normal(p.swaps_per_station_day, p.swaps_per_station_day * 0.20)
                        ),
                        "opex_per_station_month": max(
                            50,
                            rng.normal(p.opex_per_station_month, p.opex_per_station_month * 0.12),
                        ),
                        "discount_rate_annual": max(0.05, rng.normal(p.discount_rate_annual, 0.02)),
                    }
                )
                npvs.append(sim.dcf()["npv"])
                revs.append(sim.pl_projection()["revenue"].sum())
            except Exception:
                continue

        npvs_arr = np.array(npvs)
        revs_arr = np.array(revs)

        return {
            "npv_simulations": npvs_arr,
            "revenue_simulations": revs_arr,
            "npv_p10": round(float(np.percentile(npvs_arr, 10)), 0),
            "npv_p50": round(float(np.percentile(npvs_arr, 50)), 0),
            "npv_p90": round(float(np.percentile(npvs_arr, 90)), 0),
            "npv_mean": round(float(npvs_arr.mean()), 0),
            "prob_positive_npv": round(float((npvs_arr > 0).mean() * 100), 1),
            "var_95": round(float(np.percentile(npvs_arr, 5)), 0),
            "rev_p10": round(float(np.percentile(revs_arr, 10)), 0),
            "rev_p50": round(float(np.percentile(revs_arr, 50)), 0),
            "rev_p90": round(float(np.percentile(revs_arr, 90)), 0),
        }

    # ── Scenario comparison ───────────────────────────────────────────────────

    def scenarios(self) -> pd.DataFrame:
        """
        Run Base / Bull / Bear scenarios and return a comparison table.

        Returns
        -------
        pd.DataFrame
            One row per scenario with key financial metrics.
        """
        rows = []
        for name, overrides in SCENARIO_OVERRIDES.items():
            sim = FinancialModel(**{**self._config_dict(), **overrides})
            ue = sim.unit_economics()
            be = sim.breakeven()
            dcf = sim.dcf()
            pl = sim.pl_projection()
            rows.append(
                {
                    "Scenario": name,
                    "Swap Price ($)": sim.config.swap_price_usd,
                    "Swaps/Day": sim.config.swaps_per_station_day,
                    "Revenue/Stn ($)": ue["gross_revenue"],
                    "EBITDA/Stn ($)": ue["ebitda"],
                    "EBITDA Margin (%)": ue["ebitda_margin_pct"],
                    "Breakeven Swaps/Day": be["breakeven_swaps_per_day"],
                    "Payback (months)": be["payback_months"],
                    "NPV ($)": dcf["npv"],
                    "IRR (%)": dcf["irr_annual_pct"],
                    "36M Net Income ($)": pl["net_income"].sum(),
                }
            )
        return pd.DataFrame(rows)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _config_dict(self) -> dict:
        return {f: getattr(self.config, f) for f in self.config.__dataclass_fields__}

    @staticmethod
    def _irr(cashflows: list, max_iter: int = 1000, tol: float = 1e-6) -> float:
        rate = 0.01
        for _ in range(max_iter):
            npv = sum(c / (1 + rate) ** t for t, c in enumerate(cashflows))
            dnpv = sum(-t * c / (1 + rate) ** (t + 1) for t, c in enumerate(cashflows))
            if abs(dnpv) < 1e-12:
                break
            new_rate = rate - npv / dnpv
            if abs(new_rate - rate) < tol:
                return new_rate
            rate = max(-0.99, new_rate)
        return rate

    def __repr__(self) -> str:
        c = self.config
        return (
            f"FinancialModel("
            f"swap_price=${c.swap_price_usd}, "
            f"swaps/day={c.swaps_per_station_day}, "
            f"stations={c.n_stations})"
        )
