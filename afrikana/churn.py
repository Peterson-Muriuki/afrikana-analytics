"""
afrikana.churn
==============
Customer churn prediction for African mobility and fintech platforms.

The ChurnScorer class trains a Gradient Boosting classifier on customer
behavioural features and scores each customer with a 0-1 churn probability.

Example
-------
>>> from afrikana.churn import ChurnScorer
>>> import pandas as pd
>>>
>>> scorer = ChurnScorer()
>>> scorer.fit(customers_df)
>>> results = scorer.score(customers_df)
>>> at_risk  = scorer.at_risk(customers_df, threshold=0.5)
>>> print(scorer.feature_importances())
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

DEFAULT_FEATURES = [
    "swap_freq_monthly",
    "last_swap_days_ago",
    "tenure_months",
    "monthly_revenue",
]

SEGMENT_MAP = {
    "commuter":  0,
    "delivery":  1,
    "logistics": 2,
    "casual":    3,
}

RISK_BINS   = [0, 0.30, 0.60, 1.01]
RISK_LABELS = ["Low", "Medium", "High"]


@dataclass
class ChurnScorerConfig:
    """Hyperparameters and feature configuration for ChurnScorer."""
    n_estimators:   int   = 200
    max_depth:      int   = 4
    learning_rate:  float = 0.05
    random_state:   int   = 42
    test_size:      float = 0.20
    features:       list  = field(default_factory=lambda: DEFAULT_FEATURES.copy())
    segment_column: str   = "segment"
    target_column:  str   = "churned"
    verbose:        bool  = False


class ChurnScorer:
    """
    Gradient Boosting churn predictor for African mobility platforms.

    Designed for EV swap station operators, boda-boda networks,
    and similar subscription/usage-based businesses.

    Parameters
    ----------
    config : ChurnScorerConfig, optional
        Model configuration. Uses sensible defaults if not provided.

    Attributes
    ----------
    model_ : GradientBoostingClassifier
        Fitted sklearn model (available after calling fit()).
    feature_cols_ : list[str]
        Feature columns used for training (available after calling fit()).
    auc_ : float
        ROC-AUC score on the held-out test set (available after calling fit()).
    is_fitted_ : bool
        True if fit() has been called successfully.
    """

    def __init__(self, config: ChurnScorerConfig | None = None):
        self.config    = config or ChurnScorerConfig()
        self.model_    = None
        self.feature_cols_ = None
        self.auc_      = None
        self.is_fitted_ = False

    def fit(self, df: pd.DataFrame) -> "ChurnScorer":
        """
        Train the churn model on a customer DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain feature columns defined in config.features plus
            the target column (default: 'churned', binary 0/1).

        Returns
        -------
        self : ChurnScorer
            Returns self to allow method chaining.

        Raises
        ------
        ValueError
            If required columns are missing from df.
        """
        df = df.copy()
        self._validate_columns(df)

        if self.config.segment_column in df.columns:
            df["_segment_enc"] = (
                df[self.config.segment_column].map(SEGMENT_MAP).fillna(0).astype(int)
            )
            feature_cols = self.config.features + ["_segment_enc"]
        else:
            feature_cols = self.config.features.copy()

        X = df[feature_cols].fillna(0)
        y = df[self.config.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        self.model_ = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
        )
        self.model_.fit(X_train, y_train)
        self.feature_cols_ = feature_cols

        proba = self.model_.predict_proba(X_test)[:, 1]
        self.auc_ = roc_auc_score(y_test, proba)

        if self.config.verbose:
            preds = self.model_.predict(X_test)
            print(f"ChurnScorer fitted — AUC: {self.auc_:.3f}")
            print(classification_report(y_test, preds))

        self.is_fitted_ = True
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score customers with churn probability and risk tier.

        Parameters
        ----------
        df : pd.DataFrame
            Customer DataFrame (same schema as used for fit()).

        Returns
        -------
        pd.DataFrame
            Original DataFrame with two added columns:
            - churn_score : float in [0, 1]
            - churn_risk  : str — "Low", "Medium", or "High"
        """
        self._check_fitted()
        df = df.copy()

        if "_segment_enc" not in df.columns and self.config.segment_column in df.columns:
            df["_segment_enc"] = (
                df[self.config.segment_column].map(SEGMENT_MAP).fillna(0).astype(int)
            )

        X = df[[c for c in self.feature_cols_ if c in df.columns]].fillna(0)
        df["churn_score"] = self.model_.predict_proba(X)[:, 1]
        df["churn_risk"]  = pd.cut(
            df["churn_score"],
            bins=RISK_BINS,
            labels=RISK_LABELS,
        )
        return df

    def at_risk(
        self,
        df: pd.DataFrame,
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """
        Return customers above the churn threshold, sorted by score descending.

        Parameters
        ----------
        df : pd.DataFrame
            Customer DataFrame.
        threshold : float
            Churn probability cutoff. Default 0.50.

        Returns
        -------
        pd.DataFrame
            Filtered, sorted DataFrame of at-risk customers.
        """
        scored = self.score(df)
        return (
            scored[scored["churn_score"] >= threshold]
            .sort_values("churn_score", ascending=False)
            .reset_index(drop=True)
        )

    def feature_importances(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame of feature importances.

        Returns
        -------
        pd.DataFrame
            Columns: feature, importance — sorted descending.
        """
        self._check_fitted()
        return (
            pd.DataFrame({
                "feature":    self.feature_cols_,
                "importance": self.model_.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def summary(self) -> dict:
        """Return a summary dict of model metadata."""
        self._check_fitted()
        return {
            "auc":            round(self.auc_, 4),
            "n_estimators":   self.config.n_estimators,
            "features":       self.feature_cols_,
            "top_feature":    self.feature_importances().iloc[0]["feature"],
        }

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.config.features if c not in df.columns]
        if missing:
            raise ValueError(
                f"ChurnScorer: missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
        if self.config.target_column not in df.columns:
            raise ValueError(
                f"ChurnScorer: target column '{self.config.target_column}' not found in DataFrame."
            )

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "ChurnScorer is not fitted. Call scorer.fit(df) before scoring."
            )

    def __repr__(self) -> str:
        status = f"fitted (AUC={self.auc_:.3f})" if self.is_fitted_ else "not fitted"
        return f"ChurnScorer(status={status}, n_estimators={self.config.n_estimators})"
