import runpy

import pytest

BASELINE = {"CHAT_MODEL": "gemini-3.5-flash-lite", "OLLAMA_CHAT_MODEL": "qwen2.5:3b",
            "EMBEDDING_MODEL": "gemini-embedding-2", "EMBEDDING_DIMENSIONS": "768",
            "CHROMA_COLLECTION": "cv_gemini_embedding_2_768_v1", "GEMINI_MAX_RPM": "10",
            "OLLAMA_NUM_CTX": "4096", "OLLAMA_KEEP_ALIVE": "30m", "EMBED_BATCH_SIZE": "20"}


def load_config(tmp_path, monkeypatch, drop=(), **overrides):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # ignore the developer's .env
    for legacy in ("GEMINI_CHAT_MODEL", "GEMINI_EMBEDDING_MODEL", "GEMINI_EMBEDDING_DIMENSIONS"):
        monkeypatch.delenv(legacy, raising=False)
    for name, setting in {**BASELINE, "DATA_DIR": str(tmp_path), **overrides}.items():
        monkeypatch.setenv(name, setting)
    for name in drop:
        monkeypatch.delenv(name)
    return runpy.run_path("src/backend/config.py")


@pytest.mark.parametrize("key,value", [
    ("CHAT_MODEL", "other"), ("CHAT_MODEL", "gemini-3.5-flsh"),  # typo
    ("CHAT_MODEL", "gemini-3.8-flash-tts"),  # not a chat model in the allowlist
    ("OLLAMA_CHAT_MODEL", "other"), ("OLLAMA_CHAT_MODEL", "llama3:8b"),
    ("OLLAMA_CHAT_MODEL", "qwen3:4b; rm -rf"),
    ("EMBEDDING_MODEL", "gemini-embedding-001"), ("EMBEDDING_MODEL", "text-embedding-ada-002"),
    ("EMBEDDING_DIMENSIONS", "1536"), ("CHROMA_COLLECTION", "other"),
    ("GEMINI_MAX_RPM", "11"), ("OLLAMA_NUM_CTX", "8192"),
    ("OLLAMA_KEEP_ALIVE", "5m"), ("EMBED_BATCH_SIZE", "21"),
])
def test_baseline_rejects_overrides(key, value, tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        load_config(tmp_path, monkeypatch, **{key: value})


@pytest.mark.parametrize("model,provider", [
    ("gemini-3.5-flash", "gemini"), ("gemini-3.5-flash-lite", "gemini"),
    ("gemini-3.6-flash", "gemini"), ("gemini-3.7-flash", "gemini"), ("gemini-3.8-flash", "gemini"),
    ("deepseek-flash", "deepseek"), ("gpt-6-luna", "openai"), ("qwen3.8-27b", "vllm"),
    ("qwen3.5-2b", "vllm"), ("qwen3.5-4b", "vllm"),
])
def test_allowlisted_cloud_models_are_accepted(model, provider, tmp_path, monkeypatch):
    cfg = load_config(tmp_path, monkeypatch, CHAT_MODEL=model)
    assert cfg["CHAT_MODEL"] == model and cfg["CHAT_PROVIDER"] == provider


@pytest.mark.parametrize("tag", ["qwen2.5:3b", "qwen3:4b", "qwen3:8b", "qwen3-coder:30b", "qwen3"])
def test_any_qwen_tag_is_accepted_locally(tag, tmp_path, monkeypatch):
    assert load_config(tmp_path, monkeypatch, OLLAMA_CHAT_MODEL=tag)["OLLAMA_CHAT_MODEL"] == tag


def test_legacy_gemini_chat_model_env_still_works(tmp_path, monkeypatch):
    cfg = load_config(tmp_path, monkeypatch, drop=("CHAT_MODEL",), GEMINI_CHAT_MODEL="gemini-3.8-flash")
    assert cfg["CHAT_MODEL"] == "gemini-3.8-flash"


@pytest.mark.parametrize("model,dims,collection", [
    ("gemini-embedding-2", 768, "cv_gemini_embedding_2_768_v1"),
    ("text-embedding-3-small", 1536, "cv_text_embedding_3_small_1536_v1"),
    ("text-embedding-3-large", 3072, "cv_text_embedding_3_large_3072_v1"),
])
def test_each_embedding_model_gets_its_own_collection(model, dims, collection, tmp_path, monkeypatch):
    cfg = load_config(tmp_path, monkeypatch, drop=("EMBEDDING_DIMENSIONS", "CHROMA_COLLECTION"),
                      EMBEDDING_MODEL=model)
    assert (cfg["EMBEDDING_DIMENSIONS"], cfg["CHROMA_COLLECTION"]) == (dims, collection)


def test_embedding_model_cannot_reuse_another_models_collection(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        load_config(tmp_path, monkeypatch, drop=("EMBEDDING_DIMENSIONS",),
                    EMBEDDING_MODEL="text-embedding-3-small")  # baseline collection is Gemini's


@pytest.mark.parametrize("model,rpm", [("gemini-3.5-flash-lite", 10), ("deepseek-flash", None),
                                       ("gpt-6-luna", None), ("qwen3.8-27b", None)])
def test_only_gemini_is_paced(model, rpm, tmp_path, monkeypatch):
    assert load_config(tmp_path, monkeypatch, CHAT_MODEL=model)["CHAT_RPM"] == rpm
