import runpy

import pytest


@pytest.mark.parametrize("key,value", [
    ("GEMINI_CHAT_MODEL", "other"), ("OLLAMA_CHAT_MODEL", "other"),
    ("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
    ("GEMINI_EMBEDDING_DIMENSIONS", "1536"), ("CHROMA_COLLECTION", "other"),
    ("GEMINI_MAX_RPM", "13"), ("OLLAMA_NUM_CTX", "8192"),
    ("OLLAMA_KEEP_ALIVE", "5m"), ("EMBED_BATCH_SIZE", "21"),
])
def test_baseline_rejects_overrides(key, value, tmp_path, monkeypatch):
    values = {"GEMINI_CHAT_MODEL": "gemini-3.5-flash-lite", "OLLAMA_CHAT_MODEL": "qwen2.5:3b",
              "GEMINI_EMBEDDING_MODEL": "gemini-embedding-2", "GEMINI_EMBEDDING_DIMENSIONS": "768",
              "CHROMA_COLLECTION": "cv_gemini_embedding_2_768_v1", "GEMINI_MAX_RPM": "12",
              "OLLAMA_NUM_CTX": "4096", "OLLAMA_KEEP_ALIVE": "30m", "EMBED_BATCH_SIZE": "20",
              "DATA_DIR": str(tmp_path)}
    values[key] = value
    for name, setting in values.items():
        monkeypatch.setenv(name, setting)
    with pytest.raises(ValueError):
        runpy.run_path("src/backend/config.py")
