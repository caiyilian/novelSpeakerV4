import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_label  # noqa: E402


class QuoteScopeHintTests(unittest.TestCase):
    def test_non_colon_intervening_attribution_defaults_to_previous_quote(self):
        hint = run_label.Boss._format_structure_hint(
            None,
            {
                "previous": (10, "上一句"),
                "next": (13, "下一句"),
                "narrative_between": [(11, "角色甲向角色乙搭腔。")],
                "no_narrative_break_from_previous": False,
                "has_nearby_attribution_word": True,
            },
        )

        self.assertIn("PREVIOUS quote, not TARGET", hint)

    def test_colon_intervening_attribution_may_introduce_target(self):
        hint = run_label.Boss._format_structure_hint(
            None,
            {
                "previous": (10, "上一句"),
                "next": (13, "下一句"),
                "narrative_between": [(11, "角色甲回答：")],
                "no_narrative_break_from_previous": False,
                "has_nearby_attribution_word": True,
            },
        )

        self.assertIn("may introduce TARGET", hint)

    def test_comparative_phrase_is_not_extracted_as_a_speaker(self):
        candidates = run_label._extract_attribution_candidates(
            "与其说来人像商人，不如说更像贵族。"
        )

        self.assertEqual([], candidates)

    def test_dialogue_contact_verb_is_recognized(self):
        candidates = run_label._extract_attribution_candidates("角色甲向角色乙搭腔。")

        self.assertEqual("角色甲", candidates[0]["speaker"])
        self.assertEqual("搭腔", candidates[0]["verb"])

    def test_inability_to_speak_is_not_an_attribution(self):
        candidates = run_label._extract_attribution_candidates("角色甲说不出话来，只能低下头。")

        self.assertEqual([], candidates)

    def test_inverted_contact_attribution_uses_post_verbal_subject(self):
        candidates = run_label._extract_attribution_candidates("向角色乙搭腔的是角色甲。")

        self.assertEqual([{"speaker": "角色甲", "verb": "搭腔"}], candidates)

    def test_long_contact_clause_keeps_leading_subject(self):
        candidates = run_label._extract_attribution_candidates(
            "角色甲也发现了角色乙，他先向旁人招呼几句后，便向角色乙搭腔。"
        )

        self.assertIn({"speaker": "角色甲", "verb": "搭腔"}, candidates)

    def test_rhetorical_question_is_not_extracted_as_a_speaker(self):
        candidates = run_label._extract_attribution_candidates("这还用说吗？")

        self.assertEqual([], candidates)

    def test_descriptive_inability_is_not_extracted_as_a_speaker(self):
        candidates = run_label._extract_attribution_candidates("这种心情无法说清。")

        self.assertEqual([], candidates)


class ValidationIdentityTests(unittest.TestCase):
    def test_verified_role_answer_may_upgrade_to_name(self):
        identities = {
            "村民": {"角色甲"},
            "角色甲": {"角色甲"},
        }

        matched, reason = run_label._validation_lenient_match(
            {"村民"}, {"角色甲"}, verified_identities=identities
        )

        self.assertTrue(matched)
        self.assertEqual("verified-alias", reason)

    def test_verified_name_answer_cannot_downgrade_to_role(self):
        identities = {
            "村民": {"角色甲"},
            "角色甲": {"角色甲"},
        }

        matched, _ = run_label._validation_lenient_match(
            {"角色甲"}, {"村民"}, verified_identities=identities
        )

        self.assertFalse(matched)

    def test_organization_role_may_use_more_or_less_specific_generic_label(self):
        matched, reason = run_label._validation_lenient_match(
            {"某商行的人"}, {"商行的人"}
        )

        self.assertTrue(matched)
        self.assertEqual("generic-contained", reason)

    def test_generic_answer_may_upgrade_to_contained_personal_name(self):
        matched, reason = run_label._validation_lenient_match(
            {"角色甲商行老板"}, {"角色甲"}
        )

        self.assertTrue(matched)
        self.assertEqual("named-upgrade-contained", reason)


if __name__ == "__main__":
    unittest.main()
