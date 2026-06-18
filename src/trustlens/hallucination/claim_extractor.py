from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


OPINION_PREFIXES = (
    "i think", "i believe", "in my opinion", "perhaps", "maybe",
    "probably", "it seems", "might be",
)

QUESTION_PATTERN = re.compile(r"\?\s*$")


@dataclass
class Claim:
    text: str
    index: int
    is_factual: bool = True


@dataclass
class ClaimExtractionResult:
    claims: list[Claim] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": [{"text": c.text, "index": c.index} for c in self.claims],
            "skipped": self.skipped,
        }


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _is_factual_claim(sentence: str) -> bool:
    lower = sentence.lower().strip()
    if QUESTION_PATTERN.search(lower):
        return False
    if any(lower.startswith(p) for p in OPINION_PREFIXES):
        return False
    if len(lower.split()) < 4:
        return False
    return True


class ClaimExtractor:
    def extract(self, response: str) -> ClaimExtractionResult:
        sentences = _split_sentences(response)
        claims: list[Claim] = []
        skipped: list[str] = []

        for i, sent in enumerate(sentences):
            if _is_factual_claim(sent):
                claims.append(Claim(text=sent, index=i))
            else:
                skipped.append(sent)

        return ClaimExtractionResult(claims=claims, skipped=skipped)
