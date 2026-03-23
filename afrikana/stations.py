"""
afrikana.stations
=================
Swap station deployment optimizer for African mobility networks.

Scores candidate locations using a weighted multi-criteria model:
  - Demand density (population × commuter index)
  - Coverage gap (distance to nearest existing station)
  - Revenue potential (income index × population)
  - Underserved score (gap / density ratio)

Example
-------
>>> from afrikana.stations import StationOptimizer
>>>
>>> optimizer = StationOptimizer()
>>> candidates = optimizer.generate_grid(center=(-1.286389, 36.817223), n=40)
>>> scored     = optimizer.score(candidates, existing_stations_df)
>>> top5       = optimizer.recommend(scored, top_n=5)
>>> print(top5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist


@dataclass
class OptimizerConfig:
    """Weights for the multi-criteria scoring model. Must sum to 1.0."""
    weight_demand_density:    float = 0.30
    weight_coverage_gap:      float = 0.25
    weight_revenue_potential: float = 0.25
    weight_underserved:       float = 0.20
    coverage_radius_km:       float = 2.0

    def __post_init__(self):
        total = (
            self.weight_demand_density
            + self.weight_coverage_gap
            + self.weight_revenue_potential
            + self.weight_underserved
        )
        if not (0.99 < total < 1.01):
            raise ValueError(f"OptimizerConfig weights must sum to 1.0, got {total:.3f}")


class StationOptimizer:
    """
    Multi-criteria swap station deployment optimizer.

    Scores candidate locations on demand density, coverage gap,
    revenue potential, and underserved population index.

    Parameters
    ----------
    config : OptimizerConfig, optional
        Scoring weights. Defaults to balanced weights summing to 1.0.

    Example
    -------
    >>> opt = StationOptimizer()
    >>> candidates = opt.generate_grid((-1.286, 36.817), n=50)
    >>> scored     = opt.score(candidates, existing_df)
    >>> print(opt.recommend(scored))
    """

    def __init__(self, config: OptimizerConfig | None = None):
        self.config = config or OptimizerConfig()

    def generate_grid(
        self,
        center: tuple[float, float],
        n: int = 40,
        spread: float = 0.08,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate synthetic candidate locations around a city centre.

        Used when you don't have real candidate data — creates plausible
        grid points with synthetic demand proxies for demonstration.

        Parameters
        ----------
        center : (lat, lon)
            City centre coordinates.
        n : int
            Number of candidate locations to generate. Default 40.
        spread : float
            Degree spread around the centre. Default 0.08 (~9km).
        random_seed : int
            NumPy seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Columns: candidate_id, lat, lon, pop_density, commuter_index, income_index.
        """
        rng  = np.random.default_rng(random_seed)
        lat0, lon0 = center
        return pd.DataFrame({
            "candidate_id":   [f"CND{i+1:03d}" for i in range(n)],
            "lat":            lat0 + rng.uniform(-spread, spread, n),
            "lon":            lon0 + rng.uniform(-spread, spread, n),
            "pop_density":    rng.uniform(500, 8000, n),
            "commuter_index": rng.uniform(0.2, 1.0, n),
            "income_index":   rng.uniform(0.3, 0.9, n),
        })

    def score(
        self,
        candidates: pd.DataFrame,
        existing_stations: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score candidate locations against existing station coverage.

        Parameters
        ----------
        candidates : pd.DataFrame
            Must contain: lat, lon, pop_density, commuter_index, income_index.
        existing_stations : pd.DataFrame
            Must contain: lat, lon, status. Only active stations are used.

        Returns
        -------
        pd.DataFrame
            candidates with added columns:
            - nearest_station_km : distance to closest existing active station
            - demand_density, coverage_gap, revenue_potential, underserved_score : normalised [0,1]
            - deployment_score   : weighted composite score [0, 100]
            - priority           : "High", "Medium", or "Low"
        """
        df     = candidates.copy()
        active = existing_stations[existing_stations.get("status", "active") == "active"][["lat", "lon"]].values

        if len(active) > 0:
            grid = df[["lat", "lon"]].values
            dists = cdist(
                np.radians(grid), np.radians(active),
                metric=lambda u, v: 6371 * np.sqrt(
                    (u[0] - v[0]) ** 2
                    + ((u[1] - v[1]) * np.cos((u[0] + v[0]) / 2)) ** 2
                ),
            )
            df["nearest_station_km"] = dists.min(axis=1)
        else:
            df["nearest_station_km"] = 999.0

        def norm(s):
            r = s.max() - s.min()
            return (s - s.min()) / r if r > 0 else s * 0 + 0.5

        cfg = self.config
        df["demand_density"]    = norm(df["pop_density"] * df["commuter_index"])
        df["coverage_gap"]      = norm(df["nearest_station_km"])
        df["revenue_potential"] = norm(df["income_index"] * df["pop_density"])
        df["underserved_score"] = norm(df["nearest_station_km"] / (df["pop_density"] + 1) * 1000)

        df["deployment_score"] = (
            cfg.weight_demand_density    * df["demand_density"]
            + cfg.weight_coverage_gap   * df["coverage_gap"]
            + cfg.weight_revenue_potential * df["revenue_potential"]
            + cfg.weight_underserved    * df["underserved_score"]
        ) * 100

        df["deployment_score"] = df["deployment_score"].round(1)
        df["priority"]         = pd.cut(
            df["deployment_score"],
            bins=[0, 40, 65, 101],
            labels=["Low", "Medium", "High"],
        )
        return df.sort_values("deployment_score", ascending=False).reset_index(drop=True)

    def recommend(self, scored: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Return the top N deployment recommendations.

        Parameters
        ----------
        scored : pd.DataFrame
            Output of score().
        top_n : int
            Number of top sites to return. Default 5.

        Returns
        -------
        pd.DataFrame
            Top N rows with key scoring columns.
        """
        cols = [c for c in
                ["candidate_id", "lat", "lon", "deployment_score", "priority",
                 "nearest_station_km", "demand_density", "coverage_gap"]
                if c in scored.columns]
        return scored[cols].head(top_n).reset_index(drop=True)

    def coverage_stats(self, scored: pd.DataFrame) -> dict:
        """
        Summary statistics of the scoring results.

        Returns
        -------
        dict
            Keys: total_candidates, high_priority, medium_priority, low_priority,
            avg_coverage_gap_km, avg_score.
        """
        return {
            "total_candidates":  len(scored),
            "high_priority":     int((scored["priority"] == "High").sum()),
            "medium_priority":   int((scored["priority"] == "Medium").sum()),
            "low_priority":      int((scored["priority"] == "Low").sum()),
            "avg_coverage_gap_km": round(float(scored["nearest_station_km"].mean()), 2),
            "avg_score":         round(float(scored["deployment_score"].mean()), 1),
        }

    def __repr__(self) -> str:
        c = self.config
        return (
            f"StationOptimizer("
            f"demand={c.weight_demand_density}, "
            f"gap={c.weight_coverage_gap}, "
            f"revenue={c.weight_revenue_potential}, "
            f"underserved={c.weight_underserved})"
        )
