"""Check dependencies and official endpoints without printing credentials."""

import importlib.util
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def load_environment():
    path = Path(".env")
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def gemini_api_key():
    return os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()


def probe(url, headers=None):
    try:
        request = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(request, timeout=10) as response:
            return {"status": response.status, "reachable": True}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "reachable": True}
    except (OSError, urllib.error.URLError) as exc:
        return {"reachable": False, "error": str(exc)}


def main():
    load_environment()
    names = ("fastapi", "langgraph", "langchain_google_genai", "chromadb", "pypdf", "dotenv", "pytest", "httpx")
    installed = {name: importlib.util.find_spec(name) is not None for name in names}
    key = gemini_api_key()
    model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash").removeprefix("models/")
    from urllib.parse import quote
    report = {"dependencies_installed": installed,
              "api_key_configured": bool(key),
              "pypi": probe("https://pypi.org/simple/fastapi/"),
              "gemini": probe("https://generativelanguage.googleapis.com/v1beta/models/" + quote(model, safe=""), {"x-goog-api-key": key})
                        if key else {"reachable": False, "error": "GEMINI_API_KEY or GOOGLE_API_KEY is not configured"}}
    path = Path("data/benchmarks/runtime-check.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0 if all(installed.values()) and report["gemini"].get("status") == 200 else 2


if __name__ == "__main__":
    raise SystemExit(main())
