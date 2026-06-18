"""Regression tests for medical hallucination trust score calibration."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HALLUCINATED_MEDICAL = {
    "prompt": "Summarize the patient chart for the care team.",
    "response": (
        "The patient was diagnosed with Type 2 diabetes in 2019 and is currently "
        "on metformin 500mg twice daily. According to a 2022 study [14], 87% of "
        "similar patients require insulin within 6 months."
    ),
    "context": (
        "Patient chart: 45-year-old with prediabetes, lifestyle counseling recommended. "
        "No diabetes diagnosis. No medications prescribed."
    ),
}

GROUNDED_MEDICAL = {
    "prompt": "Summarize the patient chart for the care team.",
    "response": (
        "The patient is a 45-year-old with prediabetes. Lifestyle counseling has "
        "been recommended and no medications are currently prescribed."
    ),
    "context": HALLUCINATED_MEDICAL["context"],
}


@pytest.fixture
def pipeline():
    from trustlens.pipeline import TrustLensPipeline

    return TrustLensPipeline()


def test_medical_hallucination_low_trust(pipeline):
    result = pipeline.audit(**HALLUCINATED_MEDICAL)
    assert result.trust.domain == "medical"
    assert result.hallucination.contradicted >= 2
    assert result.trust.trust_score <= 25, (
        f"Fully hallucinated medical summary should score <=25, got {result.trust.trust_score}"
    )


def test_medical_hallucination_caps_applied(pipeline):
    result = pipeline.audit(**HALLUCINATED_MEDICAL)
    assert result.trust.caps_applied is not None
    assert len(result.trust.caps_applied) >= 1


def test_grounded_medical_high_trust(pipeline):
    result = pipeline.audit(**GROUNDED_MEDICAL)
    assert result.trust.trust_score >= 65, (
        f"Grounded medical summary should score >=65, got {result.trust.trust_score}"
    )


def test_grounded_beats_hallucinated_medical(pipeline):
    bad = pipeline.audit(**HALLUCINATED_MEDICAL)
    good = pipeline.audit(**GROUNDED_MEDICAL)
    assert good.trust.trust_score > bad.trust.trust_score + 40
