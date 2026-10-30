import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
APP_DB_PATH = Path(os.getenv("APP_DB_PATH", DATA_DIR / "app.db"))
CHECKPOINT_DB_PATH = Path(os.getenv("CHECKPOINT_DB_PATH", DATA_DIR / "checkpoints.db"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))
# ---- Fixed dual-model baseline (Rule 1.1) ----
# Exactly one cloud chat model, one local chat model and one embedding model.
PRIMARY_CHAT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_CHAT_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768

GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", PRIMARY_CHAT_MODEL).removeprefix("models/")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", EMBEDDING_MODEL).removeprefix("models/")
GEMINI_EMBEDDING_DIMENSIONS = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", str(EMBEDDING_DIMENSIONS)))
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "1"))
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", FALLBACK_CHAT_MODEL)
_default_collection = f"cv_{GEMINI_EMBEDDING_MODEL.replace('-', '_')}_{GEMINI_EMBEDDING_DIMENSIONS}_v1"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", _default_collection)

if GEMINI_CHAT_MODEL != PRIMARY_CHAT_MODEL:
    raise ValueError(f"GEMINI_CHAT_MODEL is pinned to {PRIMARY_CHAT_MODEL}")
if OLLAMA_CHAT_MODEL != FALLBACK_CHAT_MODEL:
    raise ValueError(f"OLLAMA_CHAT_MODEL is pinned to {FALLBACK_CHAT_MODEL}")
if GEMINI_EMBEDDING_MODEL != EMBEDDING_MODEL or GEMINI_EMBEDDING_DIMENSIONS != EMBEDDING_DIMENSIONS:
    raise ValueError(f"Embeddings are pinned to {EMBEDDING_MODEL} with {EMBEDDING_DIMENSIONS} dimensions")
if CHROMA_COLLECTION != "cv_gemini_embedding_2_768_v1":
    raise ValueError("CHROMA_COLLECTION is pinned to cv_gemini_embedding_2_768_v1")
if GEMINI_TIMEOUT_SECONDS <= 0 or GEMINI_MAX_RETRIES < 0:
    raise ValueError("Gemini timeout must be positive and retries nonnegative")

# ---- Local fallback (Rule 1.2): keep qwen2.5:3b resident, bounded context ----
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
TEMPERATURE = 0.0

# ---- Token / pacing budget (Rules 2.x, 3.x) ----
GEMINI_MAX_RPM = int(os.getenv("GEMINI_MAX_RPM", "12"))
# Bound input size for both providers; actual token counts depend on their tokenizers.
LLM_INPUT_MAX_CHARS = int(os.getenv("LLM_INPUT_MAX_CHARS", "8000"))
EMBED_TEXT_MAX_WORDS = 300
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "20"))
EMBED_BACKOFF_SECONDS = (10, 20, 40)
EMBED_BACKOFF_JITTER = (1.0, 3.0)
RECOMMENDED_BATCH_LIMIT = (20, 50)

if not 1 <= GEMINI_MAX_RPM <= 12:
    raise ValueError("GEMINI_MAX_RPM must be 1..12")
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
MAX_EXTRACT_ATTEMPTS = 2
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
