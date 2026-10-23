"""Offline HTTP doubles; these tests do not constitute real Gemini evidence."""

import io
import json
import socket
import unittest
import urllib.error
from unittest.mock import patch

from scripts.check_gemini_model import check_model


class GeminiModelCheckTests(unittest.TestCase):
    def test_success_records_usage_and_sends_key_only_as_header(self):
        response = io.BytesIO(json.dumps({"responseId": "fixture-response", "modelVersion": "fixture-model",
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "MODEL_OK"}]}}],
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3, "totalTokenCount": 11}}).encode())
        response.status = 200
        response.headers = {"x-request-id": "fixture-request-id"}
        with patch("scripts.check_gemini_model.urllib.request.urlopen", return_value=response) as call:
            report = check_model("gemini-3.8-flash", "test-only-secret")
        request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "Reply with exactly MODEL_OK")
        self.assertNotIn("test-only-secret", request.full_url)
        self.assertEqual(report["status"], "succeeded")
        self.assertEqual(report["usage"]["totalTokenCount"], 11)
        self.assertEqual(report["cv_records_sent"], 0)
        self.assertNotIn("test-only-secret", json.dumps(report))

    def test_error_message_never_persists_rejected_key(self):
        error = urllib.error.HTTPError("https://generativelanguage.googleapis.com/", 400, "Bad request", {},
            io.BytesIO(json.dumps({"error": {"message": "Invalid key: test-only-secret",
                "status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}]}}).encode()))
        with patch("scripts.check_gemini_model.urllib.request.urlopen", side_effect=error):
            report = check_model("gemini-3.8-flash", "test-only-secret")
        self.assertEqual(report["http_status"], 400)
        self.assertEqual(report["api_error_reason"], "API_KEY_INVALID")
        self.assertNotIn("test-only-secret", json.dumps(report))

    def test_dns_missing_key_and_incomplete_response_are_not_success(self):
        with patch("scripts.check_gemini_model.urllib.request.urlopen", side_effect=urllib.error.URLError(socket.gaierror(11001, "DNS failed"))):
            self.assertEqual(check_model("gemini-3.8-flash", "test-key")["status"], "network_error")
        with patch("scripts.check_gemini_model.urllib.request.urlopen") as call:
            self.assertEqual(check_model("gemini-3.8-flash", "")["status"], "missing_api_key")
            call.assert_not_called()
        response = io.BytesIO(json.dumps({"modelVersion": "fixture-model", "candidates": [
            {"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "partial"}]}}]}).encode())
        response.status = 200
        response.headers = {}
        with patch("scripts.check_gemini_model.urllib.request.urlopen", return_value=response):
            self.assertEqual(check_model("gemini-3.8-flash", "test-key")["status"], "blocked_or_incomplete_response")
