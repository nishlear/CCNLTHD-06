"""SDK boundary doubles only; no real Google requests or SDK installation required."""

import json
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.check_gemini_model import check_model


class FakeAPIError(Exception):
    code = 429
    status = "RESOURCE_EXHAUSTED"


def sdk_modules(data=None, error=None):
    google = ModuleType("google")
    genai = ModuleType("google.genai")
    genai.errors = SimpleNamespace(APIError=FakeAPIError)
    genai.types = SimpleNamespace(HttpOptions=lambda **kw: SimpleNamespace(**kw),
        HttpRetryOptions=lambda **kw: SimpleNamespace(**kw),
        ThinkingConfig=lambda **kw: SimpleNamespace(**kw),
        GenerateContentConfig=lambda **kw: SimpleNamespace(**kw))
    genai.Client = MagicMock()
    call = genai.Client.return_value.__enter__.return_value.models.generate_content
    if error is not None:
        call.side_effect = error
    else:
        call.return_value.model_dump.return_value = data
    google.genai = genai
    return {"google": google, "google.genai": genai}, genai, call


def completed_response():
    return {"response_id": "fixture-response", "model_version": "fixture-model",
            "candidates": [{"finish_reason": "STOP", "content": {"parts": [{"text": "MODEL_OK"}]}}],
            "usage_metadata": {"prompt_token_count": 8, "candidates_token_count": 3, "total_token_count": 11}}


class GeminiSDKCheckTests(unittest.TestCase):
    def test_sdk_uses_explicit_key_developer_api_ms_timeout_and_no_retries(self):
        modules, genai, call = sdk_modules(completed_response())
        with patch.dict("sys.modules", modules):
            result = check_model("gemini-3.8-flash", "fixture-secret", 60, transport="sdk")
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["usage"]["total_token_count"], 11)
        self.assertEqual(result["model_requests_attempted"], 1)
        self.assertNotIn("fixture-secret", json.dumps(result))
        options = genai.Client.call_args.kwargs
        self.assertEqual(options["api_key"], "fixture-secret")
        self.assertFalse(options["vertexai"])
        self.assertEqual(options["http_options"].timeout, 60000)
        self.assertEqual(options["http_options"].retry_options.attempts, 1)
        self.assertEqual(call.call_args.kwargs["config"].thinking_config.thinking_level, "low")
        self.assertEqual(call.call_count, 1)

    def test_sdk_api_error_preserves_status_but_never_message(self):
        modules, _, call = sdk_modules(error=FakeAPIError("fixture-secret"))
        with patch.dict("sys.modules", modules):
            result = check_model("gemini-3.8-flash", "fixture-secret", transport="sdk")
        self.assertEqual(result["status"], "api_error")
        self.assertEqual(result["http_status"], 429)
        self.assertEqual(result["api_error_status"], "RESOURCE_EXHAUSTED")
        self.assertNotIn("fixture-secret", json.dumps(result))
        self.assertEqual(call.call_count, 1)

    def test_sdk_timeout_is_sanitized_and_never_retried(self):
        modules, _, call = sdk_modules(error=TimeoutError("fixture-secret"))
        with patch.dict("sys.modules", modules):
            result = check_model("gemini-3.8-flash", "fixture-secret", transport="sdk")
        self.assertEqual(result["status"], "client_error")
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertNotIn("fixture-secret", json.dumps(result))
        self.assertEqual(call.call_count, 1)

    def test_sdk_missing_dependency_does_not_attempt_inference(self):
        with patch("builtins.__import__", side_effect=ImportError("fixture-secret")):
            result = check_model("gemini-3.8-flash", "fixture-secret", transport="sdk")
        self.assertEqual(result["status"], "missing_dependency")
        self.assertEqual(result["model_requests_attempted"], 0)
        self.assertNotIn("fixture-secret", json.dumps(result))

    def test_sdk_incomplete_output_is_not_success(self):
        data = completed_response()
        data["candidates"][0]["finish_reason"] = "MAX_TOKENS"
        modules, _, _ = sdk_modules(data)
        with patch.dict("sys.modules", modules):
            result = check_model("gemini-3.8-flash", "fixture-secret", transport="sdk")
        self.assertEqual(result["status"], "blocked_or_incomplete_response")
