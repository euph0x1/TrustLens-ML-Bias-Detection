from trustlens.pipeline import TrustLensPipeline


def test_pipeline_audit_returns_dict():
    pipeline = TrustLensPipeline()
    result = pipeline.audit(
        prompt="Evaluate candidate Maria for engineering role.",
        response="Maria is a strong candidate with excellent Python skills. I recommend hiring.",
        context="Maria: 5 years Python, led 2 projects.",
    )
    data = result.to_dict()
    assert "trust" in data
    assert "bias" in data
    assert "hallucination" in data
    assert 0 <= data["trust"]["trust_score"] <= 100


def test_pipeline_detects_toxic_bias():
    pipeline = TrustLensPipeline()
    result = pipeline.audit(
        prompt="Evaluate candidate.",
        response="Women are too emotional and stupid for this job.",
        context="",
    )
    assert result.bias.bias_score > 0.3
