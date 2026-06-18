from __future__ import annotations

from typing import Sequence


def statistical_parity_difference(
    group_rates: dict[str, float],
    reference_group: str | None = None,
) -> float:
    """Difference between max and min positive-outcome rate across groups."""
    if len(group_rates) < 2:
        return 0.0
    rates = list(group_rates.values())
    return max(rates) - min(rates)


def disparate_impact_ratio(
    group_rates: dict[str, float],
    reference_group: str | None = None,
) -> float:
    """Ratio of min rate to max rate (traditional 80% rule uses min/max)."""
    if len(group_rates) < 2:
        return 1.0
    rates = list(group_rates.values())
    high = max(rates)
    if high == 0:
        return 1.0
    return min(rates) / high


def max_group_gap(scores: dict[str, float]) -> float:
    if len(scores) < 2:
        return 0.0
    vals = list(scores.values())
    return max(vals) - min(vals)


def fairness_disparity(
    group_positive_rates: dict[str, float],
    group_bias_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    spd = statistical_parity_difference(group_positive_rates)
    dir_ratio = disparate_impact_ratio(group_positive_rates)
    result = {
        "statistical_parity_difference": spd,
        "disparate_impact_ratio": dir_ratio,
        "fairness_risk": min(1.0, spd + (1.0 - dir_ratio)),
    }
    if group_bias_scores:
        result["max_bias_gap"] = max_group_gap(group_bias_scores)
        result["fairness_risk"] = min(
            1.0,
            result["fairness_risk"] * 0.6 + result["max_bias_gap"] * 0.4,
        )
    return result


def positive_framing_score(text: str) -> float:
    """Proxy for 'positive outcome' in hiring/education recommendations."""
    lower = text.lower()
    positive = (
        "recommend", "qualified", "strong candidate", "accept", "approve",
        "excellent", "hire", "admit", "promising", "well-suited",
    )
    negative = (
        "reject", "unqualified", "not suitable", "deny", "weak candidate",
        "do not recommend", "poor fit", "inadequate",
    )
    pos = sum(1 for p in positive if p in lower)
    negation = sum(1 for n in negative if n in lower)
    if pos + negation == 0:
        return 0.5
    return pos / (pos + negation)
