from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trustlens.config import ROOT
from trustlens.fairness.metrics import fairness_disparity, positive_framing_score

DEFAULT_BENCHMARK_PATH = ROOT / "data" / "benchmarks" / "counterfactual_prompts.csv"


@dataclass
class CounterfactualScenario:
    scenario_id: str
    domain: str
    protected_attribute: str
    group_a: str
    group_b: str
    prompt_a: str
    prompt_b: str
    description: str


@dataclass
class CounterfactualResult:
    scenario_id: str
    group_scores: dict[str, float]
    group_bias_scores: dict[str, float]
    disparity: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "group_scores": {k: round(v, 4) for k, v in self.group_scores.items()},
            "group_bias_scores": {k: round(v, 4) for k, v in self.group_bias_scores.items()},
            "disparity": {k: round(v, 4) if isinstance(v, float) else v for k, v in self.disparity.items()},
        }


def load_scenarios(path: Path | None = None) -> list[CounterfactualScenario]:
    csv_path = path or DEFAULT_BENCHMARK_PATH
    scenarios: list[CounterfactualScenario] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append(
                CounterfactualScenario(
                    scenario_id=row["scenario_id"],
                    domain=row["domain"],
                    protected_attribute=row["protected_attribute"],
                    group_a=row["group_a"],
                    group_b=row["group_b"],
                    prompt_a=row["prompt_a"],
                    prompt_b=row["prompt_b"],
                    description=row["description"],
                )
            )
    return scenarios


class CounterfactualEvaluator:
    def evaluate_pair(
        self,
        scenario_id: str,
        response_a: str,
        response_b: str,
        group_a: str,
        group_b: str,
        bias_scores: dict[str, float] | None = None,
    ) -> CounterfactualResult:
        rate_a = positive_framing_score(response_a)
        rate_b = positive_framing_score(response_b)
        group_scores = {group_a: rate_a, group_b: rate_b}
        group_bias = bias_scores or {group_a: 0.0, group_b: 0.0}
        disparity = fairness_disparity(group_scores, group_bias)
        return CounterfactualResult(
            scenario_id=scenario_id,
            group_scores=group_scores,
            group_bias_scores=group_bias,
            disparity=disparity,
        )

    def evaluate_multi(
        self,
        responses: dict[str, str],
        bias_scores: dict[str, float] | None = None,
        scenario_id: str = "custom",
    ) -> CounterfactualResult:
        group_scores = {g: positive_framing_score(r) for g, r in responses.items()}
        group_bias = bias_scores or {g: 0.0 for g in responses}
        disparity = fairness_disparity(group_scores, group_bias)
        return CounterfactualResult(
            scenario_id=scenario_id,
            group_scores=group_scores,
            group_bias_scores=group_bias,
            disparity=disparity,
        )
