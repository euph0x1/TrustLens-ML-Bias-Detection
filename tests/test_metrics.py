import pytest

from trustlens.fairness.metrics import (
    disparate_impact_ratio,
    fairness_disparity,
    positive_framing_score,
    statistical_parity_difference,
)


def test_statistical_parity_difference():
    rates = {"group_a": 0.8, "group_b": 0.4}
    assert statistical_parity_difference(rates) == pytest.approx(0.4)


def test_disparate_impact_ratio():
    rates = {"group_a": 0.8, "group_b": 0.4}
    assert disparate_impact_ratio(rates) == pytest.approx(0.5)


def test_disparate_impact_single_group():
    assert disparate_impact_ratio({"a": 0.5}) == 1.0


def test_positive_framing_recommend():
    text = "I strongly recommend hiring this qualified candidate."
    assert positive_framing_score(text) > 0.5


def test_positive_framing_reject():
    text = "I reject this unqualified candidate and do not recommend."
    assert positive_framing_score(text) < 0.5


def test_fairness_disparity_includes_bias_gap():
    rates = {"a": 0.9, "b": 0.3}
    bias = {"a": 0.1, "b": 0.6}
    result = fairness_disparity(rates, bias)
    assert result["statistical_parity_difference"] == pytest.approx(0.6)
    assert result["max_bias_gap"] == pytest.approx(0.5)
    assert 0 <= result["fairness_risk"] <= 1.0
