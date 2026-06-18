from __future__ import annotations

MEDICAL_KEYWORDS = {
    "patient", "diagnosis", "medication", "medications", "chart", "clinical",
    "prediabetes", "diabetes", "prescribed", "treatment", "hospital", "doctor",
    "care team", "symptom", "medical", "physician", "nurse", "dosage", "therapy",
}

HIRING_KEYWORDS = {
    "candidate", "hire", "hiring", "interview", "resume", "job", "role",
    "applicant", "employment", "scholarship", "promote", "workforce",
}


def detect_domain(prompt: str, context: str = "") -> str:
    """Return domain label: medical, hiring, or general."""
    text = f"{prompt} {context}".lower()
    medical_hits = sum(1 for kw in MEDICAL_KEYWORDS if kw in text)
    hiring_hits = sum(1 for kw in HIRING_KEYWORDS if kw in text)
    if medical_hits >= 2 or (medical_hits >= 1 and "patient" in text):
        return "medical"
    if hiring_hits >= 1:
        return "hiring"
    return "general"
