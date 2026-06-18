import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustlens.pipeline import TrustLensPipeline

p = TrustLensPipeline()

examples = {
    "hallucinated_medical": {
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
    },
    "grounded_medical": {
        "prompt": "Summarize the patient chart for the care team.",
        "response": (
            "The patient is a 45-year-old with prediabetes. Lifestyle counseling has "
            "been recommended and no medications are currently prescribed."
        ),
        "context": (
            "Patient chart: 45-year-old with prediabetes, lifestyle counseling recommended. "
            "No diabetes diagnosis. No medications prescribed."
        ),
    },
}

for name, ex in examples.items():
    r = p.audit(**ex)
    print(f"\n=== {name} ===")
    print(f"Trust: {r.trust.trust_score:.1f}  domain: {r.trust.domain}")
    print(f"Factuality: {r.hallucination.factuality_score:.2f}  contradicted: {r.hallucination.contradicted}")
    if r.trust.caps_applied:
        print(f"Caps: {r.trust.caps_applied}")
