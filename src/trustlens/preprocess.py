from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CleanText:
    prompt: str
    response: str
    context: str


def _normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess(prompt: str, response: str, context: str = "") -> CleanText:
    return CleanText(
        prompt=_normalize(prompt),
        response=_normalize(response),
        context=_normalize(context),
    )
