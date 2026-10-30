import unittest

from src.backend.scoring import score_cv


class ScoringTests(unittest.TestCase):
    def test_report_examples(self):
        criteria = {"skills": ["Python", "FastAPI"], "min_years": 3}
        a = score_cv("a", {"skills": ["Python", "FastAPI"], "years_experience": 4}, criteria, 1)
        b = score_cv("b", {"skills": ["Python"], "years_experience": None}, criteria, .9)
        self.assertEqual((a["score"], b["score"]), (1.0, .52))
        self.assertEqual(b["missing_skills"], ["fastapi"])

    def test_case_whitespace_and_duplicate_skills(self):
        result = score_cv("a", {"skills": [" PYTHON "]},
                          {"skills": ["Python", "python", " "]}, 1)
        self.assertEqual(result["matched_skills"], ["python"])
        self.assertEqual(result["score"], 1)

    def test_unknown_experience_is_not_written_back(self):
        cv = {"skills": [], "years_experience": None}
        self.assertEqual(score_cv("a", cv, {"min_years": 3}, 1)["score"], .8)
        self.assertIsNone(cv["years_experience"])

    def test_experience_edge_cases(self):
        for years, minimum, expected in [(None, 0, 1), (None, None, 1),
                                          (None, 3, .8), (4, 3, 1), (-1, 3, .8)]:
            with self.subTest(years=years, minimum=minimum):
                result = score_cv("a", {"skills": [], "years_experience": years},
                                  {"min_years": minimum}, 1)
                self.assertEqual(result["score"], expected)

    def test_similarity_bounds(self):
        self.assertEqual(score_cv("a", {}, {}, -10)["score"], .7)
        self.assertEqual(score_cv("a", {}, {}, 10)["score"], 1)
