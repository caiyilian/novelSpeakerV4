import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dialogue_ensemble import (  # noqa: E402
    build_dialogue_packet,
    citations_are_local,
    format_dialogue_packet,
    parse_line_citations,
)
import requests  # noqa: E402

from run_label import ApiModel, QualityEnsemble, ShortMemAgent, _should_failover_to_next_model  # noqa: E402


class DialoguePacketTests(unittest.TestCase):
    def setUp(self):
        quote_open = "\u300c"
        quote_close = "\u300d"
        self.lines = [
            "Narrative introduction.",
            f"{quote_open}first{quote_close}",
            "Narrative attribution.",
            f"{quote_open}second{quote_close}",
            f"{quote_open}third{quote_close}",
            "Closing narrative.",
        ]
        self.dialogues = [(2, "first"), (4, "second"), (5, "third")]

    def test_packet_marks_the_exact_target_turn(self):
        packet = build_dialogue_packet(self.lines, self.dialogues, 4, "second", dialogue_radius=1)

        target_turns = [turn for turn in packet["turns"] if turn["is_target"]]
        self.assertEqual(len(target_turns), 1)
        self.assertEqual(target_turns[0]["text"], "second")
        self.assertEqual(packet["target"]["same_line_ordinal"], 1)
        self.assertIn("TARGET", format_dialogue_packet(packet))

    def test_only_packet_lines_are_accepted_as_citations(self):
        packet = build_dialogue_packet(self.lines, self.dialogues, 4, "second", dialogue_radius=1)

        self.assertEqual(parse_line_citations("L2, L4 and L4"), [2, 4])
        self.assertTrue(citations_are_local([4], packet))
        self.assertFalse(citations_are_local([99], packet))

    def test_truncation_keeps_the_target_raw_line_visible(self):
        lines = ["x" * 500 for _ in range(20)]
        lines[10] = "target raw text"
        packet = build_dialogue_packet(lines, [(3, "a"), (11, "target"), (18, "b")], 11, "target", dialogue_radius=1)

        rendered = format_dialogue_packet(packet, max_raw_chars=250)

        self.assertIn(">> L11: target raw text", rendered)


class QualityDecisionTests(unittest.TestCase):
    def setUp(self):
        self.packet = {
            "raw_lines": [
                {"line": 10, "text": "raw"},
                {"line": 11, "text": "raw"},
            ]
        }
        self.ensemble = QualityEnsemble(dialogue_radius=1)

    def test_weak_arbiter_cannot_override_cited_blind_consensus(self):
        quote_card = {"voice_kind": "person", "quote_type": "direct_speech"}
        local_card = {"speaker": "Alpha", "citations": [10]}
        block_card = {"speaker": "Alpha", "citations": [11]}
        arbiter_card = {
            "speaker": "Beta",
            "citations": [10],
            "confidence": "medium",
            "evidence_basis": "inference",
            "quote_type": "direct_speech",
        }

        decision = self.ensemble._select_decision(
            quote_card, local_card, block_card, arbiter_card, self.packet
        )

        self.assertEqual(decision["speaker"], "Alpha")
        self.assertTrue(decision["blind_agreement"])
        self.assertTrue(decision["confirmed_anchor"])

    def test_only_confirmed_memory_can_supply_an_exchange_hint(self):
        memory = ShortMemAgent(max_rounds=8)
        memory.update(1, "a", "Alpha", confirmed=True)
        memory.update(2, "b", "Beta", confirmed=True)
        memory.update(3, "c", "Alpha", confirmed=True)
        memory.update(4, "d", "Beta", confirmed=True)
        self.assertEqual(memory.get_next_expected(), "Alpha")

        memory = ShortMemAgent(max_rounds=8)
        memory.update(1, "a", "Alpha", confirmed=False)
        memory.update(2, "b", "Beta", confirmed=False)
        memory.update(3, "c", "Alpha", confirmed=False)
        memory.update(4, "d", "Beta", confirmed=False)
        self.assertIsNone(memory.get_next_expected())

    def test_each_quality_role_has_its_own_required_tool_fields(self):
        tool = self.ensemble._quality_output_tool(("target_speaker", "citations"))

        self.assertEqual(
            tool["function"]["parameters"]["required"],
            ["target_speaker", "citations"],
        )


class ApiFallbackTests(unittest.TestCase):
    def test_round_robin_timeout_fails_over_instead_of_retrying_same_key(self):
        pooled = ApiModel(
            "pooled",
            "model",
            "https://example.invalid",
            "key",
            round_robin_group="provider-pool",
        )
        standalone = ApiModel(
            "standalone",
            "model",
            "https://example.invalid",
            "key",
        )

        timeout = requests.exceptions.ReadTimeout("timed out")
        self.assertTrue(_should_failover_to_next_model(timeout, pooled))
        self.assertFalse(_should_failover_to_next_model(timeout, standalone))


if __name__ == "__main__":
    unittest.main()
