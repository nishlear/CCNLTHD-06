"""Make one real Gemini generateContent request and save safe evidence; no CVs."""

import argparse
import json
import math
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from scripts.check_runtime import gemini_api_key, load_environment

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"


def check_model(model, key, timeout_seconds=60.0, transport="urllib"):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        return {"status": "invalid_timeout", "model_requests_attempted": 0, "cv_records_sent": 0}
    if transport not in {"urllib", "sdk"}:
        return {"status": "invalid_transport", "model_requests_attempted": 0, "cv_records_sent": 0}
    model = model.removeprefix("models/")
    if not re.fullmatch(r"gemini-[a-zA-Z0-9._-]+", model):
        return {"status": "invalid_model_name", "model_requests_attempted": 0, "cv_records_sent": 0}
    endpoint = BASE_URL + model + ":generateContent"
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "provider": "google_gemini", "endpoint": endpoint, "requested_model": model,
              "timeout_seconds": timeout_seconds, "transport": transport,
              "model_requests_attempted": 0, "cv_records_sent": 0,
              "scope": "One short inference check, not a CV workflow or embedding evaluation."}
    if not key:
        return {**report, "status": "missing_api_key"}
    if transport == "sdk":
        started = perf_counter()
        check_sdk(model, key, timeout_seconds, report)
        report["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        return report
    generation = {"maxOutputTokens": 1024}
    if model.startswith("gemini-3"):
        generation["thinkingConfig"] = {"thinkingLevel": "low"}
    body = {"contents": [{"role": "user", "parts": [{"text": "Reply with exactly MODEL_OK"}]}],
            "generationConfig": generation}
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                     headers={"x-goog-api-key": key,
                                              "Content-Type": "application/json"}, method="POST")
    report["model_requests_attempted"] = 1
    started = perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            report["http_status"] = response.status
            report["request_id"] = response.headers.get("x-request-id")
            data = json.load(response)
        candidates = data.get("candidates", [])
        first = candidates[0] if candidates else {}
        text = "".join(part.get("text", "") for part in first.get("content", {}).get("parts", [])
                       if not part.get("thought", False))
        valid = first.get("finishReason") == "STOP" and bool(text) and bool(data.get("modelVersion"))
        report.update(status="succeeded" if valid else "blocked_or_incomplete_response",
                      response_id=data.get("responseId"), returned_model=data.get("modelVersion"),
                      finish_reason=first.get("finishReason"), output_text=text[:200],
                      expected_output_match=text.strip() == "MODEL_OK", usage=data.get("usageMetadata"))
    except urllib.error.HTTPError as exc:
        report.update(status="api_error", http_status=exc.code)
        try:
            error = json.loads(exc.read()).get("error", {})
            # Google errors can echo credentials. Persist only bounded enum codes.
            for label, value in (("api_error_status", error.get("status")),
                                 ("api_error_reason", next((x.get("reason") for x in error.get("details", [])
                                                           if isinstance(x, dict) and x.get("reason")), None))):
                if isinstance(value, str) and re.fullmatch(r"[A-Z0-9_]{1,64}", value):
                    report[label] = value
        except (ValueError, OSError, AttributeError, TypeError):
            pass
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        report.update(status="network_error", error_type=type(reason).__name__)
        number = getattr(reason, "errno", None)
        if isinstance(number, int):
            report["os_error"] = number
    except (ValueError, TypeError, AttributeError, IndexError):
        report["status"] = "invalid_response"
    report["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
    return report



def check_sdk(model, key, timeout_seconds, report):
    """Check the application's official SDK without exposing exception messages."""
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError:
        report.update(status="missing_dependency", dependency="google-genai")
        return
    try:
        options = types.HttpOptions(api_version="v1beta", timeout=max(1, int(timeout_seconds * 1000)),
                                    retry_options=types.HttpRetryOptions(attempts=1))
        generation = {"max_output_tokens": 1024}
        if model.startswith("gemini-3"):
            generation["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        with genai.Client(api_key=key, vertexai=False, http_options=options) as client:
            report["model_requests_attempted"] = 1
            response = client.models.generate_content(
                model=model, contents="Reply with exactly MODEL_OK",
                config=types.GenerateContentConfig(**generation))
            data = response.model_dump(mode="json", exclude_none=True)
        candidates = data.get("candidates", [])
        first = candidates[0] if candidates else {}
        text = "".join(part.get("text", "") for part in first.get("content", {}).get("parts", [])
                       if not part.get("thought", False))
        finish = first.get("finish_reason", first.get("finishReason"))
        version = data.get("model_version", data.get("modelVersion"))
        report.update(status="succeeded" if finish == "STOP" and text and version else "blocked_or_incomplete_response",
                      returned_model=version, response_id=data.get("response_id", data.get("responseId")),
                      finish_reason=finish, output_text=text[:200], expected_output_match=text.strip() == "MODEL_OK",
                      usage=data.get("usage_metadata", data.get("usageMetadata")))
    except errors.APIError as exc:
        report.update(status="api_error", error_type=type(exc).__name__)
        if isinstance(getattr(exc, "code", None), int):
            report["http_status"] = exc.code
        value = getattr(exc, "status", None)
        if isinstance(value, str) and re.fullmatch(r"[A-Z0-9_]{1,64}", value):
            report["api_error_status"] = value
    except Exception as exc:
        # Includes SDK/httpx timeouts and configuration errors. Never persist str(exc).
        report.update(status="client_error", error_type=type(exc).__name__)


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"))
    parser.add_argument("--transport", choices=("urllib", "sdk"), default="urllib",
                        help="stdlib REST check or installed official Google SDK; no automatic retries")
    parser.add_argument("--report", type=Path, default=Path("data/benchmarks/gemini-model-check.json"))
    parser.add_argument("--timeout", type=float, default=os.getenv("GEMINI_TIMEOUT_SECONDS", "60"),
                        help="Socket timeout in seconds; defaults to GEMINI_TIMEOUT_SECONDS or 60")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and greater than zero")
    report = check_model(args.model, gemini_api_key(), timeout_seconds=args.timeout, transport=args.transport)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    partial = args.report.with_suffix(".json.part")
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(args.report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["status"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
