from trustlens.hallucination.claim_extractor import ClaimExtractor


def test_extracts_factual_claims():
    extractor = ClaimExtractor()
    response = (
        "The patient has prediabetes and receives lifestyle counseling. "
        "I think maybe we should wait. "
        "What about medication?"
    )
    result = extractor.extract(response)
    assert len(result.claims) == 1
    assert "prediabetes" in result.claims[0].text
    assert len(result.skipped) == 2


def test_skips_short_sentences():
    extractor = ClaimExtractor()
    result = extractor.extract("Yes. No way.")
    assert len(result.claims) == 0
