"""Compatibility entry point; the project now exclusively uses Gemini."""

import sys
from scripts.check_gemini_model import check_model, main

if __name__ == "__main__":
    print("Use python -m scripts.check_gemini_model; this command now checks Gemini.", file=sys.stderr)
    raise SystemExit(main())
