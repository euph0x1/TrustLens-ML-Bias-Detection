from __future__ import annotations

import logging
from typing import Any

import numpy as np

from trustlens.config import load_config

logger = logging.getLogger(__name__)

_TOXIC_TERMS = {
    "idiot", "stupid", "hate", "kill", "worthless", "inferior",
    "subhuman", "trash", "disgusting", "filthy",
}

# cross-encoder/nli-deberta-v3-small label order
NLI_LABELS = ("contradiction", "entailment", "neutral")


class ModelRegistry:
    """Lazy-loaded ML models with heuristic fallbacks for offline/tests."""

    _toxicity_pipeline = None
    _nli_model = None
    _similarity_model = None
    _spacy_nlp = None
    _config: dict | None = None

    @classmethod
    def _cfg(cls) -> dict:
        if cls._config is None:
            cls._config = load_config()
        return cls._config

    @classmethod
    def use_ml(cls) -> bool:
        return bool(cls._cfg().get("models", {}).get("use_ml_models", True))

    @classmethod
    def get_toxicity_pipeline(cls):
        if cls._toxicity_pipeline is not None:
            return cls._toxicity_pipeline
        if not cls.use_ml():
            return None
        try:
            from transformers import pipeline

            model_name = cls._cfg()["models"]["toxicity"]
            cls._toxicity_pipeline = pipeline(
                "text-classification",
                model=model_name,
                top_k=None,
                truncation=True,
            )
            logger.info("Loaded toxicity model: %s", model_name)
        except Exception as exc:
            logger.warning("Toxicity model unavailable, using heuristics: %s", exc)
            cls._toxicity_pipeline = None
        return cls._toxicity_pipeline

    @classmethod
    def get_nli_model(cls):
        if cls._nli_model is not None:
            return cls._nli_model
        if not cls.use_ml():
            return None
        try:
            from sentence_transformers import CrossEncoder

            model_name = cls._cfg()["models"]["nli"]
            cls._nli_model = CrossEncoder(model_name)
            logger.info("Loaded NLI model: %s", model_name)
        except Exception as exc:
            logger.warning("NLI model unavailable, using heuristics: %s", exc)
            cls._nli_model = None
        return cls._nli_model

    @classmethod
    def get_similarity_model(cls):
        if cls._similarity_model is not None:
            return cls._similarity_model
        if not cls.use_ml():
            return None
        try:
            from sentence_transformers import SentenceTransformer

            model_name = cls._cfg()["models"]["similarity"]
            cls._similarity_model = SentenceTransformer(model_name)
            logger.info("Loaded similarity model: %s", model_name)
        except Exception as exc:
            logger.warning("Similarity model unavailable, using heuristics: %s", exc)
            cls._similarity_model = None
        return cls._similarity_model

    @classmethod
    def get_spacy(cls):
        if cls._spacy_nlp is not None:
            return cls._spacy_nlp
        try:
            import spacy

            model_name = cls._cfg()["models"]["spacy"]
            cls._spacy_nlp = spacy.load(model_name)
        except Exception as exc:
            logger.warning("spaCy unavailable: %s", exc)
            raise
        return cls._spacy_nlp

    @classmethod
    def _softmax(cls, logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    @classmethod
    def toxicity_score(cls, text: str) -> float:
        pipe = cls.get_toxicity_pipeline()
        if pipe is None:
            lower = text.lower()
            hits = sum(1 for t in _TOXIC_TERMS if t in lower)
            return min(1.0, hits * 0.35)

        results = pipe(text[:512])
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]
        toxic_labels = {"toxic", "hate", "offensive", "label_1", "1"}
        max_score = 0.0
        for item in results:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            if label in toxic_labels or "toxic" in label or "hate" in label:
                max_score = max(max_score, score)
            elif len(results) == 2 and label.endswith("1"):
                max_score = max(max_score, score)
        if max_score == 0.0 and results:
            max_score = max(float(r.get("score", 0.0)) for r in results)
        return min(1.0, max_score)

    @classmethod
    def semantic_similarity(cls, text_a: str, text_b: str) -> float:
        sim_model = cls.get_similarity_model()
        if sim_model is not None:
            emb = sim_model.encode([text_a[:512], text_b[:512]])
            sim = float(
                np.dot(emb[0], emb[1])
                / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-9)
            )
            return max(0.0, min(1.0, (sim + 1) / 2))

        a_words = set(text_a.lower().split())
        b_words = set(text_b.lower().split())
        if not a_words or not b_words:
            return 0.0
        return len(a_words & b_words) / max(len(b_words), 1)

    @classmethod
    def verify_claim(cls, premise: str, hypothesis: str) -> tuple[float, float, float]:
        """Return (entailment, contradiction, neutral) probabilities in [0, 1]."""
        nli = cls.get_nli_model()

        if nli is not None:
            logits = nli.predict([(premise[:512], hypothesis[:512])])
            probs = cls._softmax(np.asarray(logits))[0]
            contradiction = float(probs[0])
            entailment = float(probs[1])
            neutral = float(probs[2])
            return entailment, contradiction, neutral

        p_words = set(premise.lower().split())
        h_words = set(hypothesis.lower().split())
        overlap = len(p_words & h_words) / max(len(h_words), 1)
        if overlap > 0.6:
            return min(1.0, overlap), 0.1, 0.2
        if overlap < 0.15:
            return 0.1, 0.1, 0.7
        return 0.2, 0.15, 0.5

    @classmethod
    def preload(cls) -> dict[str, Any]:
        status = {}
        status["toxicity"] = cls.get_toxicity_pipeline() is not None or not cls.use_ml()
        status["nli"] = cls.get_nli_model() is not None or not cls.use_ml()
        status["similarity"] = cls.get_similarity_model() is not None or not cls.use_ml()
        try:
            cls.get_spacy()
            status["spacy"] = True
        except Exception:
            status["spacy"] = False
        return status
