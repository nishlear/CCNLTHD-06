"""Real SQLite persistence tests; only the configured database path is isolated."""

import importlib.util
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.directory.cleanup)
        settings = types.ModuleType("src.backend.config")
        settings.APP_DB_PATH = Path(self.directory.name) / "app.db"
        path = Path(__file__).resolve().parents[1] / "src/backend/db.py"
        spec = importlib.util.spec_from_file_location("src.backend._database_test", path)
        self.db = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"src.backend.config": settings}):
            spec.loader.exec_module(self.db)
        self.db.init_db()
        self.db.create_search("s", "Python", 2)
        self.db.save_shortlist("s", [
            {"cv_id": name, "score": 1, "matched_skills": ["python"],
             "missing_skills": [], "rank": rank}
            for rank, name in enumerate(("a", "b"), 1)])
        self.db.update_search("s", status="awaiting_review")

    def test_atomic_claim_across_connections(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            claims = list(pool.map(lambda _: self.db.claim_review("s"), range(4)))
        self.assertEqual(claims.count(True), 1)

    def test_questions_only_for_approved_candidates(self):
        self.db.save_review("s", ["a"], [
            {"cv_id": name, "category": "technical", "question": "Explain Python"}
            for name in ("a", "b", "outside")])
        self.assertEqual([q["cv_id"] for q in self.db.get_questions("s")], ["a"])
        self.assertEqual(self.db.get_search("s")["status"], "completed")
        self.assertEqual([r["approved"] for r in self.db.get_shortlist("s")], [True, False])

    def test_replayed_save_does_not_duplicate_questions(self):
        questions = [{"cv_id": "a", "category": "technical", "question": "Python?"}]
        self.db.save_review("s", ["a"], questions)
        self.db.save_review("s", ["a"], questions)
        self.assertEqual(len(self.db.get_questions("s")), 1)

    def test_empty_review_completes_without_questions(self):
        self.db.save_review("s", [], [])
        self.assertEqual(self.db.get_questions("s"), [])
        self.assertTrue(all(r["approved"] is False for r in self.db.get_shortlist("s")))

    def test_failed_transaction_rolls_back_approval(self):
        with self.assertRaises(KeyError):
            self.db.save_review("s", ["a"], [{"cv_id": "a", "question": "Missing category"}])
        self.assertTrue(all(r["approved"] is None for r in self.db.get_shortlist("s")))
        self.assertEqual(self.db.get_search("s")["status"], "awaiting_review")

    def test_cv_round_trip_preserves_unknown_experience(self):
        self.db.save_cv("a", "cv.pdf", "ready", extracted={"skills": ["Python"]})
        self.assertIsNone(self.db.get_cv("a")["years_experience"])
        self.assertEqual(self.db.list_cvs("failed"), [])
        self.assertTrue(self.db.delete_cv("a"))
        self.assertIsNone(self.db.get_cv("a"))

    def test_summary_pagination_has_stable_order_and_excludes_raw_text(self):
        for name in ("a", "b", "c"):
            self.db.save_cv(name, name + ".pdf", "ready", raw_text="long CV text",
                            extracted={"skills": ["Python"]})
        first = self.db.list_cvs("ready", limit=2, offset=0)
        second = self.db.list_cvs("ready", limit=2, offset=2)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 1)
        self.assertFalse({r["id"] for r in first} & {r["id"] for r in second})
        self.assertNotIn("raw_text", first[0])
