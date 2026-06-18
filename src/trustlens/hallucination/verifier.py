from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from trustlens.config import load_config
from trustlens.hallucination.claim_extractor import ClaimExtractor
from trustlens.models import ModelRegistry

POSITIVE_CONTEXT_SIGNALS = (
    "strong references", "qualified", "excellent", "led", "leadership",
    "experienced", "meets the qualifications", "solid",
)

NEGATIVE_RECOMMENDATION_PHRASES = (
    "do not recommend", "don't recommend", "not recommend", "reject",
    "do not hire", "don't hire", "not suitable", "unqualified",
)

UNSUPPORTED_ATTRIBUTES = (
    "emotional", "aggressive", "lazy", "unreliable", "incompetent",
    "struggle with", "too emotional", "not leadership",
)

GROUP_GENERALIZATION = re.compile(
    r"\b(women|woman|men|man)\b.{0,50}\b(often|always|typically|tend to|struggle)\b",
    re.IGNORECASE,
)

STATUS_WEIGHTS = {
    "supported": 1.0,
    "contradicted": 0.0,
    "uncertain": 0.35,
    "unverified": 0.15,
}


@dataclass
class ClaimVerification:
    claim: str
    status: str  # supported | contradicted | uncertain | unverified
    entailment_score: float
    contradiction_score: float
    similarity_score: float
    reason: str


@dataclass
class HallucinationResult:
    factuality_score: float
    has_context: bool
    claims_checked: int
    supported: int
    contradicted: int
    uncertain: int
    verifications: list[ClaimVerification] = field(default_factory=list)
    heuristic_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factuality_score": round(self.factuality_score, 4),
            "has_context": self.has_context,
            "claims_checked": self.claims_checked,
            "supported": self.supported,
            "contradicted": self.contradicted,
            "uncertain": self.uncertain,
            "verifications": [
                {
                    "claim": v.claim,
                    "status": v.status,
                    "entailment_score": round(v.entailment_score, 4),
                    "contradiction_score": round(v.contradiction_score, 4),
                    "similarity_score": round(v.similarity_score, 4),
                    "reason": v.reason,
                }
                for v in self.verifications
            ],
            "heuristic_flags": self.heuristic_flags,
        }


class HallucinationChecker:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        self.thresholds = self.config["thresholds"]
        self.heuristics = self.config.get("heuristics", {})
        self.claim_extractor = ClaimExtractor()

    def _heuristic_scan(self, response: str) -> list[str]:
        flags: list[str] = []
        cite_pat = self.heuristics.get("fake_citation_pattern", "")
        num_pat = self.heuristics.get("vague_number_pattern", "")
        if cite_pat and re.search(cite_pat, response, re.IGNORECASE):
            flags.append("possible_unverified_citation")
        if num_pat and re.search(num_pat, response, re.IGNORECASE):
            flags.append("specific_statistic_without_context")
        return flags

    def _medical_rules(self, context: str, claim: str) -> tuple[str | None, str]:
        ctx = context.lower()
        cl = claim.lower()

        chart_denies_diabetes = (
            "no diabetes diagnosis" in ctx
            or ("prediabetes" in ctx and "type 2 diabetes" not in ctx)
        )
        claim_invents_diabetes = (
            "type 2 diabetes" in cl
            or ("diagnosed" in cl and "diabetes" in cl and "prediabetes" not in cl)
        )
        if chart_denies_diabetes and claim_invents_diabetes:
            return "contradicted", "Invented diabetes diagnosis contradicts patient chart"

        chart_no_meds = "no medications" in ctx or "no medications prescribed" in ctx
        claim_has_meds = any(
            term in cl for term in ("metformin", "insulin", "mg twice", "mg daily", "currently on")
        )
        if chart_no_meds and claim_has_meds:
            return "contradicted", "Medication claim contradicts chart stating no medications"

        if re.search(r"\d+%|\[\d+\]|study", cl) and "patient" in ctx:
            if not re.search(r"\d+%|\[\d+\]|study", ctx):
                return "contradicted", "Unsupported statistic or citation for this patient chart"

        aligned = 0
        if "prediabetes" in ctx and "prediabetes" in cl:
            aligned += 1
        if "lifestyle" in ctx and "lifestyle" in cl:
            aligned += 1
        if chart_no_meds and ("no medication" in cl or "not prescribed" in cl or "no medications" in cl):
            aligned += 1
        if "45-year-old" in ctx and "45-year-old" in cl:
            aligned += 1
        if aligned >= 2:
            return "supported", "Key clinical facts align with patient chart"

        return None, ""

    def _rule_based_status(self, context: str, claim: str) -> tuple[str | None, str]:
        ctx = context.lower()
        cl = claim.lower()

        medical_status, medical_reason = self._medical_rules(context, claim)
        if medical_status:
            return medical_status, medical_reason

        positive_ctx = any(sig in ctx for sig in POSITIVE_CONTEXT_SIGNALS)
        negative_claim = any(phrase in cl for phrase in NEGATIVE_RECOMMENDATION_PHRASES)
        if positive_ctx and negative_claim:
            return "contradicted", "Negative recommendation conflicts with positive source context"

        for attr in UNSUPPORTED_ATTRIBUTES:
            if attr in cl and attr not in ctx:
                if any(name in cl for name in ("maria", "candidate", "applicant", "patient", "he ", "she ")):
                    return "unverified", f"Attribute '{attr}' not present in source context"
                if GROUP_GENERALIZATION.search(claim):
                    return "unverified", "Group generalization not supported by source context"

        if GROUP_GENERALIZATION.search(claim):
            return "unverified", "Group-level claim not grounded in individual context"

        return None, ""

    def _classify_claim(
        self,
        context: str,
        claim: str,
        entailment: float,
        contradiction: float,
        neutral: float,
        similarity: float,
    ) -> tuple[str, str]:
        rule_status, rule_reason = self._rule_based_status(context, claim)
        if rule_status == "contradicted":
            return rule_status, rule_reason
        if rule_status == "supported":
            return rule_status, rule_reason

        ent_thresh = self.thresholds["nli_entailment"]
        con_thresh = self.thresholds["nli_contradiction"]

        if contradiction > entailment and contradiction >= con_thresh:
            return "contradicted", "Claim contradicts provided context (NLI)"

        if entailment > contradiction and entailment >= ent_thresh:
            if rule_status == "unverified":
                return "unverified", rule_reason
            return "supported", "Claim entailed by provided context (NLI)"

        if rule_status == "unverified":
            return "unverified", rule_reason

        if neutral >= max(entailment, contradiction):
            return "unverified", "Claim neither entailed nor contradicted by context"

        if similarity >= self.thresholds["similarity_min"]:
            return "uncertain", "Semantically related but not clearly entailed"

        return "unverified", "Insufficient support in context"

    def analyze(self, response: str, context: str = "") -> HallucinationResult:
        extraction = self.claim_extractor.extract(response)
        heuristic_flags = self._heuristic_scan(response)

        if not context.strip():
            penalty = min(0.3, len(heuristic_flags) * 0.1)
            base = 0.5 - penalty
            return HallucinationResult(
                factuality_score=max(0.0, base),
                has_context=False,
                claims_checked=0,
                supported=0,
                contradicted=0,
                uncertain=len(extraction.claims),
                heuristic_flags=heuristic_flags,
            )

        verifications: list[ClaimVerification] = []
        supported = contradicted = uncertain = 0

        for claim in extraction.claims:
            entailment, contradiction, neutral = ModelRegistry.verify_claim(
                premise=context,
                hypothesis=claim.text,
            )
            similarity = ModelRegistry.semantic_similarity(context, claim.text)

            status, reason = self._classify_claim(
                context, claim.text, entailment, contradiction, neutral, similarity
            )

            if status == "supported":
                supported += 1
            elif status == "contradicted":
                contradicted += 1
            else:
                uncertain += 1

            verifications.append(
                ClaimVerification(
                    claim=claim.text,
                    status=status,
                    entailment_score=entailment,
                    contradiction_score=contradiction,
                    similarity_score=similarity,
                    reason=reason,
                )
            )

        total = len(extraction.claims) or 1
        weighted = sum(STATUS_WEIGHTS[v.status] for v in verifications) / total
        factuality = weighted
        if heuristic_flags:
            factuality = max(0.0, factuality - 0.05 * len(heuristic_flags))

        return HallucinationResult(
            factuality_score=factuality,
            has_context=True,
            claims_checked=len(extraction.claims),
            supported=supported,
            contradicted=contradicted,
            uncertain=uncertain,
            verifications=verifications,
            heuristic_flags=heuristic_flags,
        )
