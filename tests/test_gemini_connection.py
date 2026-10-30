"""Offline connection diagnostics and credential redaction checks."""

import io
import json
import subprocess
import unittest
import urllib.error
from unittest.mock import patch

from scripts.diagnose_gemini_connection import ROOT_URL, curl_probe, proxy_summary, python_probe


class ConnectionDiagnosticTests(unittest.TestCase):
    def test_python_head_404_is_a_response_without_credentials(self):
        error = urllib.error.HTTPError(ROOT_URL, 404, "Not Found", {}, io.BytesIO(b"discard"))
        with patch("scripts.diagnose_gemini_connection.urllib.request.urlopen", side_effect=error) as call:
            result = python_probe(ROOT_URL, "HEAD", 10)
        self.assertTrue(result["http_response_received"])
        self.assertEqual(result["http_status"], 404)
        request = call.call_args.args[0]
        self.assertEqual(request.get_method(), "HEAD")
        self.assertIsNone(request.data)
        self.assertEqual(request.header_items(), [])

    def test_python_post_auth_error_is_reachable_but_sends_no_key(self):
        error = urllib.error.HTTPError(ROOT_URL, 403, "Forbidden", {}, io.BytesIO(b"secret response"))
        with patch("scripts.diagnose_gemini_connection.urllib.request.urlopen", side_effect=error) as call:
            result = python_probe(ROOT_URL, "POST", 10)
        request = call.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.header_items(), [("Content-type", "application/json")])
        self.assertNotIn("secret response", json.dumps(result))
        self.assertEqual(result["http_status"], 403)

    def test_python_timeout_message_is_not_persisted(self):
        with patch("scripts.diagnose_gemini_connection.urllib.request.urlopen",
                   side_effect=TimeoutError("must-not-be-logged")):
            result = python_probe(ROOT_URL, "HEAD", 10)
        self.assertFalse(result["http_response_received"])
        self.assertNotIn("must-not-be-logged", json.dumps(result))

    def test_curl_post_uses_stdin_and_discards_response_and_errors(self):
        completed = subprocess.CompletedProcess([], 0, b"403 0.01 0.02 0.03", b"must-not-be-logged")
        with patch("scripts.diagnose_gemini_connection.shutil.which", return_value="curl"), patch(
                "scripts.diagnose_gemini_connection.subprocess.run", return_value=completed) as call:
            result = curl_probe(ROOT_URL, "POST", 10)
        command = call.call_args.args[0]
        self.assertIn("@-", command)
        self.assertNotIn("x-goog-api-key", " ".join(command))
        self.assertNotIn("--insecure", command)
        self.assertEqual(result["http_status"], 403)
        self.assertEqual(result["tls_ready_seconds"], 0.02)
        self.assertNotIn("must-not-be-logged", json.dumps(result))

    def test_curl_timeout_is_not_reachable(self):
        completed = subprocess.CompletedProcess([], 28, b"000 0.00 0.00 0.00", b"sensitive proxy URL")
        with patch("scripts.diagnose_gemini_connection.shutil.which", return_value="curl"), patch(
                "scripts.diagnose_gemini_connection.subprocess.run", return_value=completed):
            result = curl_probe(ROOT_URL, "HEAD", 10)
        self.assertFalse(result["http_response_received"])
        self.assertEqual(result["curl_exit_code"], 28)
        self.assertNotIn("sensitive", json.dumps(result))

    def test_proxy_report_never_contains_proxy_address_or_password(self):
        with patch("scripts.diagnose_gemini_connection.urllib.request.getproxies",
                   return_value={"https": "http://user:password@private-proxy:8080"}), patch(
                "scripts.diagnose_gemini_connection.urllib.request.proxy_bypass", return_value=False):
            result = proxy_summary()
        self.assertTrue(result["https_proxy_detected"])
        self.assertNotIn("private-proxy", json.dumps(result))
        self.assertNotIn("password", json.dumps(result))
