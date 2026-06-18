from trustlens.trust.domains import detect_domain
from trustlens.trust.scorer import TrustScorer
from trustlens.bias.detector import BiasResult
from trustlens.hallucination.verifier import HallucinationResult


def test_detect_medical_domain():
    domain = detect_domain(
        "Summarize the patient chart for the care team.",
        "Patient chart: 45-year-old with prediabetes.",
    )
    assert domain == "medical"


def test_detect_hiring_domain():
    domain = detect_domain(
        "Evaluate this candidate for a software engineering role.",
        "Candidate Maria Lopez: 5 years Python.",
    )
    assert domain == "hiring"


def test_zero_factuality_caps_trust():
    scorer = TrustScorer()
    bias = BiasResult(bias_score=0.0, toxicity=0.0)
    hall = HallucinationResult(
        factuality_score=0.0,
        has_context=True,
        claims_checked=2,
        supported=0,
        contradicted=2,
        uncertain=0,
    )
    result = scorer.compute(
        bias,
        hall,
        fairness_risk=0.0,
        prompt="Summarize the patient chart for the care team.",
        context="Patient chart: prediabetes, no medications.",
    )
    assert result.trust_score <= 25
    assert result.domain == "medical"
    assert result.caps_applied is not None


def test_multiplicative_penalizes_zero_factuality():
    scorer = TrustScorer()
    bias = BiasResult(bias_score=0.0, toxicity=0.0)
    hall = HallucinationResult(
        factuality_score=0.0,
        has_context=False,
        claims_checked=0,
        supported=0,
        contradicted=0,
        uncertain=0,
    )
    result = scorer.compute(bias, hall, fairness_risk=0.0)
    assert result.trust_score < 60
