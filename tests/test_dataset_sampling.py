import unittest

from scripts.prepare_djinni import sample_records


def profile(index, text=None):
    return {"id": str(index), "CV_lang": "en", "Position": "Developer",
            "CV": text or (f"Profile {index}: " + "Python developer experience. " * 5),
            "Experience Years": None, "Primary Keyword": "Python"}


class DatasetSamplingTests(unittest.TestCase):
    def test_duplicates_and_invalid_rows_are_excluded(self):
        a = profile(1)
        b = profile(2, a["CV"].upper())
        c = profile(3)
        short = profile(4, "short")
        out, counts = sample_records([a, b, c, short], 10, 42)
        self.assertEqual([r["source_id"] for r in out], ["1", "3"])
        self.assertEqual(counts["duplicate_id_or_cv_text"], 1)
        self.assertEqual(counts["invalid_text_length"], 1)
        self.assertTrue(all(r["display_name"].startswith("Anonymous candidate") for r in out))

    def test_reservoir_sampling_is_reproducible(self):
        rows = [profile(i) for i in range(50)]
        first, _ = sample_records(rows, 10, 42)
        second, _ = sample_records(rows, 10, 42)
        self.assertEqual(first, second)
        self.assertEqual(len({r["source_id"] for r in first}), 10)
