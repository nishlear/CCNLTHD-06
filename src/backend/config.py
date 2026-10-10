import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
APP_DB_PATH = Path(os.getenv("APP_DB_PATH", DATA_DIR / "app.db"))
CHECKPOINT_DB_PATH = Path(os.getenv("CHECKPOINT_DB_PATH", DATA_DIR / "checkpoints.db"))

RETRIEVE_K = 20
DEFAULT_TOP_K = 10
MAX_TOP_K = 20
EXPLAIN_THRESHOLD = 0.5

for d in (DATA_DIR, UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)
