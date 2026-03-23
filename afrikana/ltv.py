"""
afrikana.ltv
============
Customer Lifetime Value modelling for African mobility platforms.

Uses a discounted survival-adjusted revenue projection:
  LTV = monthly_revenue × gross_margin × expected_lifetime_months × discount_factor

Example
-------
>>> from afrikana.ltv import LTVCalculator
>>>
>>> calc = LTVCalculator(gross_margin=0.62, discount_rate_annual=0.12)
>>> results = calc.compute(customers_df)
>>> print(results[["customer_id", "ltv", "ltv_tier"]].head())
>>> print(calc.country_summary(results))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class LTVConfig:
    """Configuration for LTV calculation."""

    gross_margin: float = 0.62
    discount_rate_annual: float = 0.12
    max_horizon_months: int = 36
    churn_col: str = "churn_probability"
    revenue_col: str = "monthly_revenue"
    tenure_col: str = "tenure_months"


class LTVCalculator:
    """
    Customer Lifetime Value calculator for African mobility platforms.

    Computes discounted LTV using a survival-adjusted projection
    and segments customers into Bronze / Silver / Gold tiers.

    Parameters
    ----------
    gross_margin : float
        Gross margin percentage (0–1). Default 0.62 (62%).
    discount_rate_annual : float
        Annual discount rate (0–1). Default 0.12 (12%).
    max_horizon_months : int
        Maximum projection horizon in months. Default 36.

    Example
    -------
    >>> calc = LTVCalculator(gross_margin=0.55)
    >>> df_with_ltv = calc.compute(customers_df)
    >>> print(calc.tier_summary(df_with_ltv))
    """

    def __init__(
        self,
        gross_margin: float = 0.62,
        discount_rate_annual: float = 0.12,
        max_horizon_months: int = 36,
    ):
        self.config = LTVConfig(
            gross_margin=gross_margin,
            discount_rate_annual=discount_rate_annual,
            max_horizon_months=max_horizon_months,
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute LTV for every customer in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns for churn probability, monthly revenue,
            and tenure months (names configurable via LTVConfig).

        Returns
        -------
        pd.DataFrame
            Original DataFrame with added columns:
            - monthly_survival   : 1 - churn_probability
            - expected_lifetime  : projected remaining months
            - ltv                : discounted lifetime value in USD
            - ltv_tier           : "Bronze", "Silver", or "Gold"
        """
        cfg = self.config
        df = df.copy()

        churn_col = cfg.churn_col if cfg.churn_col in df.columns else "churn_score"
        revenue_col = cfg.revenue_col
        tenure_col = cfg.tenure_col

        if churn_col not in df.columns:
            df[churn_col] = 0.15

        r_monthly = (1 + cfg.discount_rate_annual) ** (1 / 12) - 1

        df["monthly_survival"] = 1 - df[churn_col].clip(0.01, 0.99)
        df["expected_lifetime"] = (df["monthly_survival"] / (1 - df["monthly_survival"])).clip(
            1, cfg.max_horizon_months
        )

        tenure = df[tenure_col].clip(0, cfg.max_horizon_months) if tenure_col in df.columns else 0

        df["ltv"] = (
            df[revenue_col]
            * cfg.gross_margin
            * df["expected_lifetime"]
            * (1 / (1 + r_monthly)) ** tenure
        ).round(2)

        ltv_q = df["ltv"].quantile([0.33, 0.66])
        df["ltv_tier"] = pd.cut(
            df["ltv"],
            bins=[-np.inf, ltv_q[0.33], ltv_q[0.66], np.inf],
            labels=["Bronze", "Silver", "Gold"],
        )

        return df

    def tier_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarise LTV by tier.

        Parameters
        ----------
        df : pd.DataFrame
            Output of compute().

        Returns
        -------
        pd.DataFrame
            Columns: ltv_tier, count, avg_ltv, total_ltv, avg_revenue.
        """
        return (
            df.groupby("ltv_tier", observed=False)
            .agg(
                count=("ltv", "count"),
                avg_ltv=("ltv", "mean"),
                total_ltv=("ltv", "sum"),
                avg_revenue=(self.config.revenue_col, "mean"),
            )
            .round(2)
            .reset_index()
        )

    def segment_summary(self, df: pd.DataFrame, group_col: str = "segment") -> pd.DataFrame:
        """Summarise LTV by a grouping column (e.g. segment, country, city)."""
        if group_col not in df.columns:
            raise ValueError(f"Column '{group_col}' not found in DataFrame.")
        return (
            df.groupby(group_col)
            .agg(
                count=("ltv", "count"),
                avg_ltv=("ltv", "mean"),
                total_ltv=("ltv", "sum"),
            )
            .round(2)
            .reset_index()
            .sort_values("avg_ltv", ascending=False)
        )

    def country_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience wrapper — summarise LTV by country."""
        return self.segment_summary(df, group_col="country")

    def revenue_at_risk(
        self,
        df: pd.DataFrame,
        churn_threshold: float = 0.50,
    ) -> dict:
        """
        Compute the monthly revenue and total LTV at risk from churning customers.

        Parameters
        ----------
        df : pd.DataFrame
            Output of compute() — must have churn_score and ltv columns.
        churn_threshold : float
            Probability above which a customer is considered at-risk.

        Returns
        -------
        dict
            Keys: n_at_risk, monthly_revenue_at_risk, ltv_at_risk, pct_of_total_ltv.
        """
        churn_col = "churn_score" if "churn_score" in df.columns else "churn_probability"
        at_risk = df[df[churn_col] >= churn_threshold]
        return {
            "n_at_risk": len(at_risk),
            "monthly_revenue_at_risk": round(float(at_risk[self.config.revenue_col].sum()), 2),
            "ltv_at_risk": round(float(at_risk["ltv"].sum()), 2),
            "pct_of_total_ltv": round(float(at_risk["ltv"].sum() / df["ltv"].sum() * 100), 1)
            if df["ltv"].sum() > 0
            else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"LTVCalculator("
            f"gross_margin={self.config.gross_margin}, "
            f"discount_rate={self.config.discount_rate_annual}, "
            f"horizon={self.config.max_horizon_months}M)"
        )
