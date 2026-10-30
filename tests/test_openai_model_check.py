import unittest
from scripts.check_openai_model import check_model as legacy_check
from scripts.check_gemini_model import check_model

class LegacyCheckMigrationTests(unittest.TestCase):
    def test_old_command_uses_gemini(self):
        self.assertIs(legacy_check, check_model)
