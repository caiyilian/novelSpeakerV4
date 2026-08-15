import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quote_occurrence import (  # noqa: E402
    align_quote_occurrences,
    build_quote_occurrences,
    contains_false_speech_phrase,
)


class QuoteOccurrenceTests(unittest.TestCase):
    def test_multiple_quotes_on_one_line_keep_independent_scope(self):
        records = build_quote_occurrences(
            ["角色甲说：「第一句。」角色乙回答：「第二句。」角色甲点头。"]
        )

        self.assertEqual(2, len(records))
        self.assertEqual(0, records[0]["same_line_ordinal"])
        self.assertEqual(1, records[1]["same_line_ordinal"])
        self.assertEqual("角色甲说：", records[0]["scope_before"])
        self.assertEqual("角色乙回答：", records[0]["scope_after"])
        self.assertEqual("角色乙回答：", records[1]["scope_before"])
        self.assertEqual("角色甲点头。", records[1]["scope_after"])
        self.assertLess(records[0]["quote_end"], records[1]["quote_start"])

    def test_alignment_detects_divergent_extraction(self):
        with self.assertRaisesRegex(ValueError, "alignment failed"):
            align_quote_occurrences(
                ["角色甲说：「第一句。」"],
                [(1, "另一句")],
            )

    def test_rhetorical_speech_phrases_are_not_attributions(self):
        self.assertTrue(contains_false_speech_phrase("这还用说吗？"))
        self.assertTrue(contains_false_speech_phrase("他无法说出真相。"))
        self.assertFalse(contains_false_speech_phrase("角色甲回答说。"))


if __name__ == "__main__":
    unittest.main()
