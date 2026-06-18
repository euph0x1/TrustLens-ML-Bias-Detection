from __future__ import annotations

import html
import re
from typing import Iterable

from trustlens.bias.detector import FlaggedSpan


def highlight_flagged_spans(
    text: str,
    flagged_spans: Iterable[FlaggedSpan],
    *,
    default_color: str = "#ffcccc",
) -> str:
    """Return HTML with flagged sentences highlighted."""
    escaped = html.escape(text)
    sentences = flagged_spans

    for span in sentences:
        if not span.text or span.text.startswith("("):
            continue
        target = html.escape(span.text)
        if target in escaped:
            color = _color_for_reason(span.reason)
            replacement = (
                f'<mark style="background-color:{color}; padding:2px 4px; '
                f'border-radius:3px;" title="{html.escape(span.reason)} '
                f'({span.score:.2f})">{target}</mark>'
            )
            escaped = escaped.replace(target, replacement, 1)

    return f'<div style="line-height:1.6;">{escaped}</div>'


def _color_for_reason(reason: str) -> str:
    mapping = {
        "toxicity": "#ffcccc",
        "gender_stereotype": "#ffe0b2",
        "race_stereotype": "#ffe0b2",
        "age_stereotype": "#ffe0b2",
        "negative_demographic_descriptor": "#f8bbd0",
        "contradicted": "#ffcdd2",
    }
    return mapping.get(reason, "#fff9c4")


def highlight_claims(text: str, statuses: dict[str, str]) -> str:
    """Highlight claims by verification status (supported/contradicted/uncertain)."""
    escaped = html.escape(text)
    colors = {
        "supported": "#c8e6c9",
        "contradicted": "#ffcdd2",
        "uncertain": "#fff9c4",
        "unverified": "#eeeeee",
    }
    for claim, status in statuses.items():
        target = html.escape(claim)
        if target in escaped:
            color = colors.get(status, "#eeeeee")
            replacement = (
                f'<mark style="background-color:{color}; padding:2px 4px;" '
                f'title="{html.escape(status)}">{target}</mark>'
            )
            escaped = escaped.replace(target, replacement, 1)
    return f'<div style="line-height:1.6;">{escaped}</div>'
