from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trustlens.bias.detector import BiasResult
from trustlens.config import load_config
from trustlens.hallucination.verifier import HallucinationResult
from trustlens.trust.domains import detect_domain


@dataclass
class TrustScore:
    trust_score: float
    bias_component: float
    factuality_component: float
    fairness_component: float
    weights: dict[str, float]
    domain: str = "general"
    caps_applied: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "trust_score": round(self.trust_score, 2),
            "bias_component": round(self.bias_component, 4),
            "factuality_component": round(self.factuality_component, 4),
            "fairness_component": round(self.fairness_component, 4),
            "weights": self.weights,
            "domain": self.domain,
        }
        if self.caps_applied:
            result["caps_applied"] = self.caps_applied
        return result


class TrustScorer:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        self.default_weights = self.config["trust_weights"]
        self.domain_weights = self.config.get("domain_weights", {})
        self.caps_config = self.config.get("trust_caps", {})
        self.blend_additive = float(self.config.get("trust_blend_additive", 0.35))

    def _resolve_weights(self, domain: str) -> dict[str, float]:
        if domain in self.domain_weights:
            return dict(self.domain_weights[domain])
        return dict(self.default_weights)

    def _apply_caps(
        self,
        trust: float,
        bias: BiasResult,
        hallucination: HallucinationResult,
        fairness_risk: float,
    ) -> tuple[float, list[str] | None]:
        caps = self.caps_config
        applied: list[str] = []

        if (
            hallucination.has_context
            and hallucination.factuality_score <= caps.get("zero_factuality_threshold", 0.01)
            and hallucination.claims_checked > 0
        ):
            cap = caps.get("zero_factuality_with_context", 25)
            applied.append(f"zero_factuality_cap:{cap}")
            trust = min(trust, float(cap))

        if (
            hallucination.claims_checked > 0
            and hallucination.contradicted == hallucination.claims_checked
        ):
            cap = caps.get("all_claims_contradicted", 20)
            applied.append(f"all_contradicted_cap:{cap}")
            trust = min(trust, float(cap))

        low_fact_thresh = caps.get("low_factuality_threshold", 0.15)
        if (
            hallucination.has_context
            and hallucination.factuality_score < low_fact_thresh
            and hallucination.factuality_score > caps.get("zero_factuality_threshold", 0.01)
        ):
            cap = caps.get("low_factuality_cap", 35)
            applied.append(f"low_factuality_cap:{cap}")
            trust = min(trust, float(cap))

        if bias.bias_score >= caps.get("severe_bias_threshold", 0.65):
            cap = caps.get("severe_bias_cap", 30)
            applied.append(f"severe_bias_cap:{cap}")
            trust = min(trust, float(cap))

        if fairness_risk >= caps.get("severe_fairness_threshold", 0.75):
            cap = caps.get("severe_fairness_cap", 25)
            applied.append(f"severe_fairness_cap:{cap}")
            trust = min(trust, float(cap))

        return trust, applied or None

    def compute(
        self,
        bias: BiasResult,
        hallucination: HallucinationResult,
        fairness_risk: float = 0.0,
        prompt: str = "",
        context: str = "",
    ) -> TrustScore:
        domain = detect_domain(prompt, context)
        weights = self._resolve_weights(domain)
        w_b = weights["bias"]
        w_f = weights["factuality"]
        w_fair = weights["fairness"]

        bias_component = 1.0 - bias.bias_score
        factuality_component = hallucination.factuality_score
        fairness_component = 1.0 - min(1.0, fairness_risk)

        additive = (
            w_b * bias_component
            + w_f * factuality_component
            + w_fair * fairness_component
        )
        multiplicative = bias_component * factuality_component * fairness_component
        blend = self.blend_additive
        combined = blend * additive + (1.0 - blend) * multiplicative

        trust = 100.0 * combined
        trust, caps_applied = self._apply_caps(
            trust, bias, hallucination, fairness_risk
        )

        return TrustScore(
            trust_score=max(0.0, min(100.0, trust)),
            bias_component=bias_component,
            factuality_component=factuality_component,
            fairness_component=fairness_component,
            weights=weights,
            domain=domain,
            caps_applied=caps_applied or None,
        )
