import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
APP_DB_PATH = Path(os.getenv("APP_DB_PATH", DATA_DIR / "app.db"))
CHECKPOINT_DB_PATH = Path(os.getenv("CHECKPOINT_DB_PATH", DATA_DIR / "checkpoints.db"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))
# ---- Model allowlist (Rule 1.1) ----
# One cloud chat model, one local Ollama fallback and one embedding model, each
# picked from an allowlist. The gate rejects typos and models we do not want
# (e.g. too expensive). Chat prices are USD per 1M tokens (input, output);
# embedding prices are USD per 1M input tokens; None = not published / verified.
PRIMARY_CHAT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_CHAT_MODEL = "qwen2.5:3b"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
# Gemini 3.x cannot fully turn thinking off; MINIMAL is the lowest level.
_GEMINI_FAST = {"thinking_level": "MINIMAL"}
CLOUD_CHAT_MODELS = {
    "gemini-3.5-flash":      {"provider": "gemini", "price": None, **_GEMINI_FAST},
    "gemini-3.5-flash-lite": {"provider": "gemini", "price": (0.30, 2.50), **_GEMINI_FAST},
    "gemini-3.6-flash":      {"provider": "gemini", "price": (0.75, 3.75), **_GEMINI_FAST},
    "gemini-3.7-flash":      {"provider": "gemini", "price": None, **_GEMINI_FAST},
    "gemini-3.8-flash":      {"provider": "gemini", "price": (0.75, 3.75), **_GEMINI_FAST},
    # DeepSeek, OpenAI and Qwen3.8 think by default; extraction to JSON does not
    # need it, and leaving it on adds latency/cost and makes the comparison unfair.
    # DeepSeek-V4.1-Flash-1017; peak price (off-peak is half: 0.15 / 0.60)
    "deepseek-flash":        {"provider": "deepseek", "price": (0.30, 1.20),
                              "extra_body": {"thinking": {"type": "disabled"}}},
    "gpt-6-luna":            {"provider": "openai", "price": (0.10, 0.50),
                              "extra_body": {"reasoning_effort": "none"}},
    # Self-hosted vLLM (unsloth/Qwen3.8-27B-NVFP4 on one RTX 5090): no API cost.
    "qwen3.8-27b":           {"provider": "vllm", "price": (0.0, 0.0),
                              "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    # Small self-hosted Qwen3.5 models, each served by its own vLLM instance.
    # Greedy decoding (temperature 0) makes them loop until the token cap, so they
    # use Qwen's recommended non-thinking sampling with a presence penalty instead.
    # They also echo a raw JSON Schema back, so they get a plain key list ("keys").
    "qwen3.5-2b":            {"provider": "vllm", "price": (0.0, 0.0), "prompt": "keys",
                              "base_url": os.getenv("VLLM_QWEN3_5_2B_URL", "http://127.0.0.1:8024/v1"),
                              "extra_body": {"chat_template_kwargs": {"enable_thinking": False},
                                             "temperature": 1.0, "top_p": 1.0, "top_k": 20,
                                             "presence_penalty": 2.0}},
    "qwen3.5-4b":            {"provider": "vllm", "price": (0.0, 0.0), "prompt": "keys",
                              "base_url": os.getenv("VLLM_QWEN3_5_4B_URL", "http://127.0.0.1:8027/v1"),
                              "extra_body": {"chat_template_kwargs": {"enable_thinking": False},
                                             "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                             "presence_penalty": 1.5}},
}
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8022/v1").rstrip("/")
# OpenAI-compatible endpoints: provider -> (base URL, API key env var, key required)
OPENAI_COMPAT_PROVIDERS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", True),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", True),
    "vllm": (VLLM_BASE_URL, "VLLM_API_KEY", False),
}
EMBEDDING_MODELS = {
    "gemini-embedding-2":     {"provider": "gemini", "dimensions": 768, "price": None},
    "text-embedding-3-small": {"provider": "openai", "dimensions": 1536, "price": 0.02},
    "text-embedding-3-large": {"provider": "openai", "dimensions": 3072, "price": 0.13},
}
LOCAL_MODEL_FAMILY = "qwen"  # any Ollama tag of the Qwen family, e.g. qwen3:4b

# New names first; GEMINI_* names are still read for old .env files.
CHAT_MODEL = (os.getenv("CHAT_MODEL") or os.getenv("GEMINI_CHAT_MODEL")
              or PRIMARY_CHAT_MODEL).removeprefix("models/")
EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL") or os.getenv("GEMINI_EMBEDDING_MODEL")
                   or DEFAULT_EMBEDDING_MODEL).removeprefix("models/")
if CHAT_MODEL not in CLOUD_CHAT_MODELS:
    raise ValueError(f"CHAT_MODEL must be one of {sorted(CLOUD_CHAT_MODELS)}")
if EMBEDDING_MODEL not in EMBEDDING_MODELS:
    raise ValueError(f"EMBEDDING_MODEL must be one of {sorted(EMBEDDING_MODELS)}")
CHAT_PROVIDER = CLOUD_CHAT_MODELS[CHAT_MODEL]["provider"]
EMBEDDING_PROVIDER = EMBEDDING_MODELS[EMBEDDING_MODEL]["provider"]
EMBEDDING_DIMENSIONS = EMBEDDING_MODELS[EMBEDDING_MODEL]["dimensions"]
# The legacy GEMINI_EMBEDDING_DIMENSIONS only applies while the legacy model name is in use.
_dims = os.getenv("EMBEDDING_DIMENSIONS") or (
    None if os.getenv("EMBEDDING_MODEL") else os.getenv("GEMINI_EMBEDDING_DIMENSIONS"))
if _dims is not None and int(_dims) != EMBEDDING_DIMENSIONS:
    raise ValueError(f"{EMBEDDING_MODEL} is fixed to {EMBEDDING_DIMENSIONS} dimensions")
# One Chroma collection per embedding model: vectors of different models never mix.
EXPECTED_COLLECTION = f"cv_{EMBEDDING_MODEL.replace('-', '_')}_{EMBEDDING_DIMENSIONS}_v1"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", EXPECTED_COLLECTION)
if CHROMA_COLLECTION != EXPECTED_COLLECTION:
    raise ValueError(f"CHROMA_COLLECTION must be {EXPECTED_COLLECTION} for {EMBEDDING_MODEL}")

GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))  # all cloud providers
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "1"))
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", FALLBACK_CHAT_MODEL)
if not (OLLAMA_CHAT_MODEL.split(":")[0].startswith(LOCAL_MODEL_FAMILY)
        and OLLAMA_CHAT_MODEL.replace(":", "").replace(".", "").replace("-", "").isalnum()):
    raise ValueError(f"OLLAMA_CHAT_MODEL must be a {LOCAL_MODEL_FAMILY} model tag, e.g. qwen3:4b")
if GEMINI_TIMEOUT_SECONDS <= 0 or GEMINI_MAX_RETRIES < 0:
    raise ValueError("Gemini timeout must be positive and retries nonnegative")

# ---- Local fallback (Rule 1.2): keep the Qwen model resident, bounded context ----
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
TEMPERATURE = 0.0
# Output cap for OpenAI-compatible providers. A real extraction is ~100-300 tokens;
# without a cap a small model stuck in a repetition loop generates up to its full
# context, which stalled the self-hosted vLLM servers.
CHAT_MAX_OUTPUT_TOKENS = int(os.getenv("CHAT_MAX_OUTPUT_TOKENS", "1024"))
if not 256 <= CHAT_MAX_OUTPUT_TOKENS <= 8192:
    raise ValueError("CHAT_MAX_OUTPUT_TOKENS must be 256..8192")

# ---- Token / pacing budget (Rules 2.x, 3.x) ----
# Gemini free tier allows ~10 RPM for Flash-Lite; going above it returns 429.
GEMINI_MAX_RPM = int(os.getenv("GEMINI_MAX_RPM", "10"))
# Pacing is per model: Gemini (free tier) is paced; DeepSeek/OpenAI have large
# limits and vLLM runs on our own GPU (graph MAX_CONCURRENCY bounds its load).
CHAT_RPM = GEMINI_MAX_RPM if CHAT_PROVIDER == "gemini" else CLOUD_CHAT_MODELS[CHAT_MODEL].get("rpm")
# Bound input size for both providers; actual token counts depend on their tokenizers.
LLM_INPUT_MAX_CHARS = int(os.getenv("LLM_INPUT_MAX_CHARS", "8000"))
EMBED_TEXT_MAX_WORDS = 300
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "20"))
EMBED_BACKOFF_SECONDS = (10, 20, 40)
EMBED_BACKOFF_JITTER = (1.0, 3.0)
RECOMMENDED_BATCH_LIMIT = (20, 50)

if not 1 <= GEMINI_MAX_RPM <= 10:
    raise ValueError("GEMINI_MAX_RPM must be 1..10")
if OLLAMA_NUM_CTX != 4096 or OLLAMA_KEEP_ALIVE not in ("30m", "-1"):
    raise ValueError("Ollama requires num_ctx=4096 and keep_alive=30m or -1")
if not 1000 <= LLM_INPUT_MAX_CHARS <= 8000 or OLLAMA_TIMEOUT_SECONDS <= 0:
    raise ValueError("LLM input cap must be 1000..8000 chars and Ollama timeout positive")
if not 1 <= EMBED_BATCH_SIZE <= 20:
    raise ValueError("EMBED_BATCH_SIZE must be 1..20")

RETRIEVE_K = 20
DEFAULT_TOP_K = 10
MAX_TOP_K = 20
EXPLAIN_THRESHOLD = 0.5
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.55"))
MAX_EXTRACT_ATTEMPTS = int(os.getenv("MAX_EXTRACT_ATTEMPTS", "2"))
if MAX_EXTRACT_ATTEMPTS < 1:
    raise ValueError("MAX_EXTRACT_ATTEMPTS must be >= 1")
RECURSION_LIMIT = 25
MAX_CONCURRENCY = 4
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_FILES = 20
MAX_PDF_PAGES = 100
MAX_CV_TEXT_LENGTH = 24000

if not 0 <= MIN_SIMILARITY <= 1:
    raise ValueError("MIN_SIMILARITY must be between 0 and 1")

for d in (DATA_DIR, UPLOAD_DIR, APP_DB_PATH.parent, CHECKPOINT_DB_PATH.parent):
    d.mkdir(parents=True, exist_ok=True)
