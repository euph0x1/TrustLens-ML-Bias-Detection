"""TrustLens: GenAI fairness and trust audit toolkit."""

__version__ = "0.1.0"

from trustlens.pipeline import TrustLensPipeline, AuditResult

__all__ = ["TrustLensPipeline", "AuditResult", "__version__"]
