from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trustlens.config import ROOT


def export_audit_json(result: dict[str, Any], output_dir: Path | None = None) -> Path:
    out_dir = output_dir or (ROOT / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"audit_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return path


def audit_to_markdown(result: dict[str, Any]) -> str:
    trust = result.get("trust", {})
    bias = result.get("bias", {})
    hall = result.get("hallucination", {})
    lines = [
        "# TrustLens Audit Report",
        "",
        f"**Generated:** {result.get('timestamp', 'N/A')}",
        "",
        "## Trust Score",
        f"- **Overall:** {trust.get('trust_score', 'N/A')} / 100",
        f"- Bias component: {trust.get('bias_component', 'N/A')}",
        f"- Factuality component: {trust.get('factuality_component', 'N/A')}",
        f"- Fairness component: {trust.get('fairness_component', 'N/A')}",
        "",
        "## Bias Analysis",
        f"- Bias score: {bias.get('bias_score', 'N/A')} (lower is better)",
        f"- Toxicity: {bias.get('toxicity', 'N/A')}",
    ]

    if bias.get("stereotype_hits"):
        lines.append(f"- Stereotype patterns: {', '.join(bias['stereotype_hits'])}")

    if bias.get("flagged_spans"):
        lines.extend(["", "### Flagged Spans"])
        for span in bias["flagged_spans"]:
            lines.append(f"- [{span['reason']}] {span['text']} (score: {span['score']})")

    lines.extend([
        "",
        "## Factuality Analysis",
        f"- Factuality score: {hall.get('factuality_score', 'N/A')}",
        f"- Context provided: {hall.get('has_context', False)}",
        f"- Claims checked: {hall.get('claims_checked', 0)}",
        f"- Supported: {hall.get('supported', 0)} | Contradicted: {hall.get('contradicted', 0)}",
    ])

    if hall.get("heuristic_flags"):
        lines.append(f"- Heuristic flags: {', '.join(hall['heuristic_flags'])}")

    if hall.get("verifications"):
        lines.extend(["", "### Claim Verifications"])
        for v in hall["verifications"]:
            lines.append(f"- **{v['status']}**: {v['claim']}")

    lines.extend([
        "",
        "## Limitations",
        "- Trust scores are decision-support tools, not safety certifications.",
        "- Open-domain factuality requires external retrieval when no context is provided.",
        "- Toxicity models may produce false positives on dialect and reclaimed language.",
        "",
    ])
    return "\n".join(lines)
