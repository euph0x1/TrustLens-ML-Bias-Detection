import pytest

from trustlens.config import load_config


@pytest.fixture(autouse=True)
def disable_ml_models(monkeypatch):
    """Use heuristic fallbacks in tests — no model downloads."""
    config = load_config()
    config = dict(config)
    config["models"] = dict(config["models"])
    config["models"]["use_ml_models"] = False
    monkeypatch.setattr("trustlens.models.ModelRegistry._config", config)
