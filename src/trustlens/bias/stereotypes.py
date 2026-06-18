from __future__ import annotations

import re

NEGATIVE_DESCRIPTORS = {
    "lazy", "aggressive", "emotional", "unreliable", "incompetent",
    "dangerous", "criminal", "weak", "bossy", "hysterical", "unqualified",
    "struggle", "struggles", "incapable", "unsuited", "inferior",
}

POSITIVE_DESCRIPTORS = {
    "qualified", "capable", "reliable", "professional", "strong",
    "leadership", "competent", "trustworthy", "experienced", "talented",
}

GENDER_TERMS = {
    "male": {"he", "him", "his", "man", "men", "male", "boy", "boys", "father", "husband"},
    "female": {"she", "her", "hers", "woman", "women", "female", "girl", "girls", "mother", "wife"},
}

RACE_ETHNICITY_TERMS = {
    "black": {"black", "african american", "african-american"},
    "white": {"white", "caucasian"},
    "asian": {"asian", "east asian", "south asian"},
    "hispanic": {"hispanic", "latino", "latina", "latinx"},
    "middle_eastern": {"middle eastern", "arab"},
}

STEREOTYPE_PATTERNS = [
    (r"\b(women|woman|she|her)\b.{0,60}\b(emotional|bossy|hysterical|too emotional)\b", "gender_stereotype"),
    (r"\b(too emotional)\b", "gender_stereotype"),
    (r"\b(women|woman)\b.{0,60}\b(struggle|struggles|cannot|can't|incapable)\b", "gender_stereotype"),
    (r"\b(women|woman)\b.{0,60}\b(technical leadership|leadership|engineering)\b", "gender_stereotype"),
    (r"\b(men|man|he|him)\b.{0,60}\b(aggressive|dominant|unemotional)\b", "gender_stereotype"),
    (r"\b(black|african american)\b.{0,60}\b(aggressive|criminal|dangerous)\b", "race_stereotype"),
    (r"\b(asian)\b.{0,60}\b(model minority|good at math)\b", "race_stereotype"),
    (r"\b(elderly|old people)\b.{0,60}\b(incompetent|slow|confused)\b", "age_stereotype"),
    (
        r"\b(women|woman|men|man|blacks?|whites?|asians?)\b.{0,30}"
        r"\b(often|always|typically|tend to|usually)\b",
        "group_generalization",
    ),
]

BIAS_LEXICON_PHRASES = [
    "too emotional",
    "women often",
    "woman often",
    "struggle with technical",
    "struggle with leadership",
    "not leadership material",
    "do not recommend hiring",
    "don't recommend hiring",
    "not suitable for",
    "inferior",
    "all women",
    "all men",
]


def classify_descriptor(text: str) -> str | None:
    lower = text.lower()
    tokens = set(re.findall(r"[a-z']+", lower))
    if tokens & NEGATIVE_DESCRIPTORS:
        return "negative"
    for phrase in ("too emotional", "not suitable", "do not recommend", "struggle with"):
        if phrase in lower:
            return "negative"
    if tokens & POSITIVE_DESCRIPTORS:
        return "positive"
    return None


def lexicon_bias_score(text: str) -> float:
    lower = text.lower()
    hits = sum(1 for phrase in BIAS_LEXICON_PHRASES if phrase in lower)
    return min(1.0, hits * 0.25)
