"""tests/test_churn.py"""

import pytest
import pandas as pd
from afrikana.churn import ChurnScorer, ChurnScorerConfig


def test_fit_returns_self(customers_df):
    scorer = ChurnScorer()
    result = scorer.fit(customers_df)
    assert result is scorer


def test_is_fitted_after_fit(customers_df):
    scorer = ChurnScorer()
    assert not scorer.is_fitted_
    scorer.fit(customers_df)
    assert scorer.is_fitted_


def test_auc_is_reasonable(customers_df):
    scorer = ChurnScorer()
    scorer.fit(customers_df)
    assert 0.5 <= scorer.auc_ <= 1.0


def test_score_adds_columns(customers_df):
    scorer = ChurnScorer().fit(customers_df)
    result = scorer.score(customers_df)
    assert "churn_score" in result.columns
    assert "churn_risk" in result.columns


def test_churn_score_in_range(customers_df):
    scorer = ChurnScorer().fit(customers_df)
    result = scorer.score(customers_df)
    assert result["churn_score"].between(0, 1).all()


def test_churn_risk_labels(customers_df):
    scorer = ChurnScorer().fit(customers_df)
    result = scorer.score(customers_df)
    valid = {"Low", "Medium", "High"}
    assert set(result["churn_risk"].dropna().unique()).issubset(valid)


def test_at_risk_threshold(customers_df):
    scorer = ChurnScorer().fit(customers_df)
    at_risk = scorer.at_risk(customers_df, threshold=0.5)
    assert (at_risk["churn_score"] >= 0.5).all()


def test_at_risk_sorted_descending(customers_df):
    scorer = ChurnScorer().fit(customers_df)
    at_risk = scorer.at_risk(customers_df)
    scores = at_risk["churn_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_feature_importances_shape(customers_df):
    scorer = ChurnScorer().fit(customers_df)
    imp = scorer.feature_importances()
    assert "feature" in imp.columns
    assert "importance" in imp.columns
    assert len(imp) == len(scorer.feature_cols_)


def test_feature_importances_sum_to_one(customers_df):
    scorer = ChurnScorer().fit(customers_df)
    imp = scorer.feature_importances()
    assert abs(imp["importance"].sum() - 1.0) < 1e-6


def test_not_fitted_raises(customers_df):
    scorer = ChurnScorer()
    with pytest.raises(RuntimeError, match="not fitted"):
        scorer.score(customers_df)


def test_missing_columns_raises():
    scorer = ChurnScorer()
    bad_df = pd.DataFrame({"customer_id": [1, 2], "name": ["a", "b"]})
    with pytest.raises(ValueError, match="missing required columns"):
        scorer.fit(bad_df)


def test_custom_config(customers_df):
    config = ChurnScorerConfig(n_estimators=50, verbose=False)
    scorer = ChurnScorer(config=config)
    scorer.fit(customers_df)
    assert scorer.config.n_estimators == 50


def test_repr(customers_df):
    scorer = ChurnScorer().fit(customers_df)
    r = repr(scorer)
    assert "ChurnScorer" in r
    assert "fitted" in r
