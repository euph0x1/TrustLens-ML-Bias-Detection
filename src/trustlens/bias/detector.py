from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from trustlens.bias.stereotypes import (
    GENDER_TERMS,
    RACE_ETHNICITY_TERMS,
    STEREOTYPE_PATTERNS,
    classify_descriptor,
    lexicon_bias_score,
)
from trustlens.config import load_config
from trustlens.models import ModelRegistry


@dataclass
class FlaggedSpan:
    text: str
    reason: str
    score: float
    start: int = -1
    end: int = -1


@dataclass
class DemographicMention:
    entity: str
    category: str
    sentence: str
    descriptor_tone: str | None
    toxicity: float


@dataclass
class BiasResult:
    bias_score: float
    toxicity: float
    flagged_spans: list[FlaggedSpan] = field(default_factory=list)
    demographic_analysis: list[DemographicMention] = field(default_factory=list)
    stereotype_hits: list[str] = field(default_factory=list)
    lexicon_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bias_score": round(self.bias_score, 4),
            "toxicity": round(self.toxicity, 4),
            "lexicon_score": round(self.lexicon_score, 4),
            "flagged_spans": [
                {"text": s.text, "reason": s.reason, "score": round(s.score, 4)}
                for s in self.flagged_spans
            ],
            "demographic_analysis": [
                {
                    "entity": d.entity,
                    "category": d.category,
                    "sentence": d.sentence,
                    "descriptor_tone": d.descriptor_tone,
                    "toxicity": round(d.toxicity, 4),
                }
                for d in self.demographic_analysis
            ],
            "stereotype_hits": self.stereotype_hits,
        }


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _find_demographic_mentions(text: str) -> list[tuple[str, str, str]]:
    mentions: list[tuple[str, str, str]] = []
    lower = text.lower()
    sentences = _split_sentences(text)

    for category, terms in GENDER_TERMS.items():
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", lower):
                for sent in sentences:
                    if re.search(rf"\b{re.escape(term)}\b", sent.lower()):
                        mentions.append((term, f"gender:{category}", sent))

    for category, terms in RACE_ETHNICITY_TERMS.items():
        for term in terms:
            if term in lower:
                for sent in sentences:
                    if term in sent.lower():
                        mentions.append((term, f"race_ethnicity:{category}", sent))

    try:
        nlp = ModelRegistry.get_spacy()
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                sent = ent.sent.text if ent.sent else text
                mentions.append((ent.text, "person:ner", sent))
    except Exception:
        pass

    seen: set[tuple[str, str, str]] = set()
    unique: list[tuple[str, str, str]] = []
    for m in mentions:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


class BiasDetector:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        self.thresholds = self.config["thresholds"]

    def analyze(self, response: str) -> BiasResult:
        if not response.strip():
            return BiasResult(bias_score=0.0, toxicity=0.0)

        toxicity = ModelRegistry.toxicity_score(response)
        lexicon = lexicon_bias_score(response)
        flagged: list[FlaggedSpan] = []
        stereotype_hits: list[str] = []

        sentences = _split_sentences(response)
        for sent in sentences:
            sent_tox = ModelRegistry.toxicity_score(sent)
            if sent_tox >= self.thresholds["bias_flag_span"]:
                flagged.append(
                    FlaggedSpan(text=sent, reason="toxicity", score=sent_tox)
                )

        lower = response.lower()
        for pattern, label in STEREOTYPE_PATTERNS:
            match = re.search(pattern, lower, re.IGNORECASE)
            if match:
                if label not in stereotype_hits:
                    stereotype_hits.append(label)
                flagged.append(
                    FlaggedSpan(
                        text=match.group(0),
                        reason=label,
                        score=0.85,
                    )
                )

        demographic_analysis: list[DemographicMention] = []
        for entity, category, sentence in _find_demographic_mentions(response):
            sent_tox = ModelRegistry.toxicity_score(sentence)
            tone = classify_descriptor(sentence)
            demographic_analysis.append(
                DemographicMention(
                    entity=entity,
                    category=category,
                    sentence=sentence,
                    descriptor_tone=tone,
                    toxicity=sent_tox,
                )
            )
            if tone == "negative":
                flagged.append(
                    FlaggedSpan(
                        text=sentence,
                        reason="negative_demographic_descriptor",
                        score=max(sent_tox, 0.75),
                    )
                )

        neg_rate = 0.0
        if demographic_analysis:
            neg_count = sum(1 for d in demographic_analysis if d.descriptor_tone == "negative")
            neg_rate = neg_count / len(demographic_analysis)

        stereotype_penalty = min(1.0, len(stereotype_hits) * 0.35)

        bias_score = min(
            1.0,
            0.15 * toxicity + 0.25 * neg_rate + 0.35 * stereotype_penalty + 0.25 * lexicon,
        )

        if stereotype_hits:
            bias_score = max(bias_score, 0.55)
        if "gender_stereotype" in stereotype_hits or "group_generalization" in stereotype_hits:
            bias_score = max(bias_score, 0.65)
        if lexicon >= 0.5:
            bias_score = max(bias_score, 0.6)

        return BiasResult(
            bias_score=bias_score,
            toxicity=toxicity,
            flagged_spans=flagged,
            demographic_analysis=demographic_analysis,
            stereotype_hits=stereotype_hits,
            lexicon_score=lexicon,
        )
