import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
APP_DB_PATH = Path(os.getenv("APP_DB_PATH", DATA_DIR / "app.db"))
CHECKPOINT_DB_PATH = Path(os.getenv("CHECKPOINT_DB_PATH", DATA_DIR / "checkpoints.db"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2").removeprefix("models/")
GEMINI_EMBEDDING_DIMENSIONS = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768"))
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "1"))
_default_collection = f"cv_{GEMINI_EMBEDDING_MODEL.replace('-', '_')}_{GEMINI_EMBEDDING_DIMENSIONS}_v1"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", _default_collection)

if GEMINI_EMBEDDING_MODEL not in {"gemini-embedding-2", "gemini-embedding-001"}:
    raise ValueError("Use gemini-embedding-2 or gemini-embedding-001 for embeddings")
if not 128 <= GEMINI_EMBEDDING_DIMENSIONS <= 3072:
    raise ValueError("GEMINI_EMBEDDING_DIMENSIONS must be 128..3072")
if GEMINI_TIMEOUT_SECONDS <= 0 or GEMINI_MAX_RETRIES < 0:
    raise ValueError("Gemini timeout must be positive and retries nonnegative")
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
