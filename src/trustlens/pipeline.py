from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from trustlens.bias.detector import BiasDetector, BiasResult
from trustlens.config import load_config
from trustlens.fairness.contextual import assess_contextual_fairness
from trustlens.fairness.counterfactual import CounterfactualEvaluator
from trustlens.hallucination.verifier import HallucinationChecker, HallucinationResult
from trustlens.preprocess import preprocess
from trustlens.trust.scorer import TrustScorer, TrustScore


@dataclass
class AuditResult:
    prompt: str
    response: str
    context: str
    bias: BiasResult
    hallucination: HallucinationResult
    trust: TrustScore
    fairness_risk: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "response": self.response,
            "context": self.context,
            "bias": self.bias.to_dict(),
            "hallucination": self.hallucination.to_dict(),
            "trust": self.trust.to_dict(),
            "fairness_risk": round(self.fairness_risk, 4),
        }


class TrustLensPipeline:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        self.bias_detector = BiasDetector(self.config)
        self.hallucination_checker = HallucinationChecker(self.config)
        self.trust_scorer = TrustScorer(self.config)
        self.counterfactual_evaluator = CounterfactualEvaluator()

    def audit(
        self,
        prompt: str,
        response: str,
        context: str = "",
        fairness_risk: float | None = None,
    ) -> AuditResult:
        cleaned = preprocess(prompt, response, context)

        bias = self.bias_detector.analyze(cleaned.response)
        hallucination = self.hallucination_checker.analyze(
            cleaned.response, cleaned.context
        )

        if fairness_risk is None:
            fairness_risk = assess_contextual_fairness(
                cleaned.prompt, cleaned.response, cleaned.context, bias
            )

        trust = self.trust_scorer.compute(
            bias,
            hallucination,
            fairness_risk,
            prompt=cleaned.prompt,
            context=cleaned.context,
        )

        return AuditResult(
            prompt=cleaned.prompt,
            response=cleaned.response,
            context=cleaned.context,
            bias=bias,
            hallucination=hallucination,
            trust=trust,
            fairness_risk=fairness_risk,
        )
