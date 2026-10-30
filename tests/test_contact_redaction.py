import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import run_dataset

from scripts.redact_contacts import redact_contacts
from scripts.run_dataset import prepare_ingest_item


class ContactRedactionTests(unittest.TestCase):
    def test_contact_channels_are_removed_without_losing_job_skills(self):
        original = (
            "Python, FastAPI, C++, C# and .NET; 5 years experience. "
            "Email: demo.person@example.org; Phone: +380 (50) 123-45-67. "
            "https://linkedin.com/in/demo-person github.com/demo-person "
            "t.me/demo_person Telegram: @demo_person "
            "Portfolio: https://portfolio.example.org/profile?id=123"
        )
        cleaned, counts = redact_contacts(original)
        for value in ("demo.person@example.org", "380", "linkedin.com", "github.com", "t.me/", "demo_person", "portfolio.example.org"):
            self.assertNotIn(value, cleaned)
        for value in ("Python", "FastAPI", "C++", "C#", ".NET", "5 years experience"):
            self.assertIn(value, cleaned)
        self.assertEqual(counts, {"email": 1, "url": 2, "profile_link": 2, "messenger": 1, "phone": 1})

    def test_plain_text_and_short_numeric_quantities_are_preserved(self):
        text = "C++ and Python 3.11; +5 years, 2020-2024; 100000 records."
        self.assertEqual(redact_contacts(text), (text, {}))

    def test_labelled_local_phone_and_filters_are_idempotent(self):
        text = "Mobile: 0901 234 567; Skype: demo.person; www.example.org/me"
        cleaned, counts = redact_contacts(text)
        self.assertNotIn("0901", cleaned)
        self.assertNotIn("demo.person", cleaned)
        self.assertEqual(counts, {"url": 1, "messenger": 1, "phone": 1})
        self.assertEqual(redact_contacts(cleaned), (cleaned, {}))

    def test_ingest_payload_uses_filtered_text_and_does_not_mutate_source(self):
        item = {"cv_id": "djinni-test", "file_name": "djinni-test.txt",
                "display_name": "Anonymous candidate test",
                "raw_text": "Python developer. Contact: demo.person@example.org",
                "source_experience_years": 11}
        payload, counts = prepare_ingest_item(item)
        self.assertNotIn("demo.person@example.org", payload["raw_text"])
        self.assertIn("demo.person@example.org", item["raw_text"])
        self.assertEqual(payload["display_name"], item["display_name"])
        self.assertNotIn("source_experience_years", payload)
        self.assertEqual(counts, {"email": 1})

    def test_real_import_loop_reprocesses_old_raw_rows_and_only_forwards_filtered_text(self):
        def record(name):
            return {"cv_id": name, "file_name": name + ".txt", "display_name": "Anonymous " + name,
                    "raw_text": "Python developer; https://github.com/demo-person"}
        records = [record(name) for name in ("old", "filtered", "new", "missing-vector")]
        filtered, _ = prepare_ingest_item(records[1])
        database = types.SimpleNamespace(init_db=Mock(), get_cv=Mock(side_effect=lambda cv_id: {
            "old": {"status": "ready", "raw_text": records[0]["raw_text"]},
            "filtered": {"status": "ready", "raw_text": filtered["raw_text"]},
            "missing-vector": {"status": "ready", "raw_text": filtered["raw_text"]},
        }.get(cv_id)))
        ingest = types.ModuleType("src.backend.graphs.ingest")
        ingest.run_ingest = Mock(side_effect=lambda items: [
            {"cv_id": item["cv_id"], "status": "ready"} for item in items])
        backend = types.ModuleType("src.backend")
        backend.db = database
        backend.services = types.SimpleNamespace(require_gemini=Mock(),
            model_settings=Mock(return_value={"provider": "google_gemini"}),
            has_vector=Mock(side_effect=lambda cv_id: cv_id == "filtered"))
        searches = types.ModuleType("src.backend.routers.searches")
        searches.create_search = Mock()
        schemas = types.ModuleType("src.backend.schemas")
        schemas.SearchRequest = Mock()
        modules = {"src.backend": backend, "src.backend.graphs.ingest": ingest,
                   "src.backend.routers.searches": searches, "src.backend.schemas": schemas}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            data = Path(directory) / "sample.jsonl"
            report = Path(directory) / "run.json"
            content = "".join(json.dumps(item) + "\n" for item in records).encode()
            data.write_bytes(content)
            data.with_suffix(".manifest.json").write_text(json.dumps({
                "output_sha256": hashlib.sha256(content).hexdigest()}), encoding="utf-8")
            argv = ["run_dataset", "--input", str(data), "--limit", "4", "--report", str(report)]
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                    patch.object(run_dataset.importlib.util, "find_spec", return_value=object()), \
                    patch.object(run_dataset.socket, "getaddrinfo", return_value=[]), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_dataset.main(), 0)
            result = json.loads(report.read_text(encoding="utf-8"))
        forwarded = ingest.run_ingest.call_args.args[0]
        self.assertEqual([item["cv_id"] for item in forwarded], ["old", "new", "missing-vector"])
        self.assertTrue(all("github.com" not in item["raw_text"] for item in forwarded))
        self.assertEqual(result["already_ready"], 1)
        self.assertEqual(result["ready"], 3)
        self.assertEqual(result["contact_redaction"]["profiles_with_redactions"], 4)
        self.assertNotIn("demo-person", json.dumps(result))
