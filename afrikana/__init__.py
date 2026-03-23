"""
afrikana-analytics
==================
Reusable analytics toolkit for African mobility, fintech,
and EV swap station networks.

Quick start
-----------
>>> from afrikana.churn import ChurnScorer
>>> from afrikana.ltv import LTVCalculator
>>> from afrikana.financial import FinancialModel
>>> from afrikana.stations import StationOptimizer
>>> from afrikana.forecast import DemandForecaster

>>> model = FinancialModel(swap_price=2.50, n_stations=20)
>>> print(model.unit_economics())
>>> print(f"Breakeven: {model.breakeven()['breakeven_swaps_per_day']} swaps/day")
"""

from afrikana.churn import ChurnScorer
from afrikana.ltv import LTVCalculator
from afrikana.financial import FinancialModel
from afrikana.stations import StationOptimizer
from afrikana.forecast import DemandForecaster
from afrikana import utils

__version__ = "0.1.0"
__author__ = "Peterson Mutegi"
__email__ = "pitmuriuki@gmail.com"

__all__ = [
    "ChurnScorer",
    "LTVCalculator",
    "FinancialModel",
    "StationOptimizer",
    "DemandForecaster",
    "utils",
]
