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
from unittest.mock import patch  # noqa: E402

import run_label  # noqa: E402
from run_label import (  # noqa: E402
    ApiModel,
    NON_PERSON_LABEL,
    QualityAudit,
    QualityEnsemble,
    ShortMemAgent,
    _should_failover_to_next_model,
    is_valid_final_speaker,
)


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


class BaselinePreservingAuditTests(unittest.TestCase):
    def setUp(self):
        self.audit = QualityAudit(dialogue_radius=4)
        self.packet = {"raw_lines": [{"line": 10, "text": "raw"}]}

    @staticmethod
    def quote_card(kind):
        return {
            "valid": True,
            "voice_kind": kind,
            "quote_type": "sound_or_text" if kind == "non_person" else "direct_speech",
            "confidence": "high",
            "citations": [10],
        }

    @staticmethod
    def speaker_card(speaker, basis="explicit_attribution"):
        return {
            "valid": True,
            "speaker": speaker,
            "evidence_basis": basis,
            "confidence": "high",
            "citations": [10],
            "quote_type": "direct_speech",
        }

    def select(self, baseline, quote_cards, attribution_cards, final_card):
        return self.audit._select_decision(
            baseline,
            {"quote_type": "unclear"},
            quote_cards,
            attribution_cards,
            final_card,
            self.packet,
        )

    def test_nonperson_requires_quote_and_attribution_support(self):
        quotes = [self.quote_card("non_person") for _ in range(3)]
        attributions = [self.speaker_card(NON_PERSON_LABEL, "quote_type") for _ in range(3)]
        decision = self.select("甲", quotes, attributions, self.speaker_card(NON_PERSON_LABEL, "quote_type"))

        self.assertEqual(decision["speaker"], NON_PERSON_LABEL)
        self.assertEqual(decision["selection_source"], "audited-non-person")

    def test_final_nonperson_alone_cannot_override_baseline(self):
        quotes = [self.quote_card("person") for _ in range(3)]
        attributions = [self.speaker_card("甲") for _ in range(3)]
        decision = self.select("甲", quotes, attributions, self.speaker_card(NON_PERSON_LABEL, "quote_type"))

        self.assertEqual(decision["speaker"], "甲")
        self.assertEqual(decision["selection_source"], "baseline-retained")

    def test_speaker_change_requires_all_scene_audits_and_final_judge(self):
        quotes = [self.quote_card("person") for _ in range(3)]
        split = [self.speaker_card("乙"), self.speaker_card("乙"), self.speaker_card("甲")]
        decision = self.select("甲", quotes, split, self.speaker_card("乙"))
        self.assertEqual(decision["speaker"], "甲")

        unanimous = [self.speaker_card("乙") for _ in range(3)]
        decision = self.select("甲", quotes, unanimous, self.speaker_card("乙"))
        self.assertEqual(decision["speaker"], "乙")
        self.assertEqual(decision["selection_source"], "audited-speaker-correction")

    def test_joint_panel_can_correct_with_two_final_and_three_scene_votes(self):
        quotes = [self.quote_card("person") for _ in range(3)]
        attributions = [self.speaker_card("甲") for _ in range(2)] + [self.speaker_card("乙") for _ in range(3)]
        final = self.speaker_card("乙")
        final.update({"panel_support": 2, "panel_size": 3})

        decision = self.select("甲", quotes, attributions, final)

        self.assertEqual(decision["speaker"], "乙")
        self.assertEqual(decision["selection_source"], "joint-panel-correction")

    def test_strong_local_panel_can_correct_named_speaker_without_final_support(self):
        quotes = [self.quote_card("person") for _ in range(3)]
        attributions = [self.speaker_card("乙") for _ in range(4)] + [self.speaker_card("甲")]
        opposing_final = self.speaker_card("甲")
        opposing_final.update({"panel_support": 3, "panel_size": 3})

        decision = self.select("甲", quotes, attributions, opposing_final)

        self.assertEqual(decision["speaker"], "乙")
        self.assertEqual(decision["selection_source"], "strong-local-panel-correction")

    def test_explicit_five_vote_attribution_can_restore_person_from_nonperson(self):
        quotes = [self.quote_card("non_person") for _ in range(3)]
        attributions = [self.speaker_card("乙") for _ in range(5)]
        final = self.speaker_card(NON_PERSON_LABEL, "quote_type")
        final.update({
            "panel_support": 2,
            "panel_size": 3,
            "panel_speakers": [NON_PERSON_LABEL, NON_PERSON_LABEL, "乙"],
        })

        decision = self.select(NON_PERSON_LABEL, quotes, attributions, final)

        self.assertEqual(decision["speaker"], "乙")
        self.assertEqual(decision["selection_source"], "explicit-local-person-restoration")

    def test_two_neighbor_anchor_votes_can_resolve_three_vote_plurality(self):
        decision = {
            "speaker": "甲",
            "baseline_changed": False,
            "selection_source": "baseline-retained",
            "confirmed_anchor": False,
        }
        previous_cards = [
            self.speaker_card("甲"),
            self.speaker_card("甲"),
            self.speaker_card("乙"),
        ]
        next_cards = [
            self.speaker_card("甲"),
            self.speaker_card("甲"),
            self.speaker_card("甲"),
        ]

        updated = self.audit._apply_split_neighbor_result(
            decision, "甲", "乙", previous_cards, next_cards
        )

        self.assertEqual(updated["speaker"], "乙")
        self.assertEqual(updated["selection_source"], "split-neighbor-anchor-correction")
        self.assertFalse(updated["confirmed_anchor"])

    def test_untranslated_and_protocol_labels_are_rejected(self):
        self.assertFalse(is_valid_final_speaker("Lawrence"))
        self.assertFalse(is_valid_final_speaker("unclear"))
        self.assertTrue(is_valid_final_speaker("甲"))
        self.assertTrue(is_valid_final_speaker(NON_PERSON_LABEL))

    def test_non_speech_reason_overrides_contradictory_person_field(self):
        quote = self.audit._normalize_card_consistency(
            {
                "voice_kind": "person",
                "quote_type": "embedded_quote",
                "reason": "This describes a facial expression, not actual speech.",
            },
            "quote",
        )
        speaker = self.audit._normalize_card_consistency(
            {
                "speaker": "甲",
                "evidence_basis": "explicit_attribution",
                "reason": "这是叙述性文字，并非人物说出的台词。",
            },
            "speaker",
        )

        self.assertEqual(quote["voice_kind"], "non_person")
        self.assertEqual(speaker["speaker"], NON_PERSON_LABEL)
        self.assertEqual(speaker["evidence_basis"], "quote_type")


class ApiFallbackTests(unittest.TestCase):
    def test_sensenova_pool_uses_every_configured_key(self):
        provider = {
            "options": {"baseURL": "https://example.invalid/v1"},
            "models": {
                run_label.SENSENOVA_MODEL: {
                    "id": run_label.SENSENOVA_MODEL,
                }
            },
        }
        keys = [f"key-{index}" for index in range(1, 8)]
        models = []

        with patch.object(
            run_label, "_load_opencode_provider", return_value=provider
        ), patch.object(run_label, "_load_sensenova_keys", return_value=keys):
            run_label._append_sensenova_models(models)

        self.assertEqual(
            [f"sense-nova-{index}" for index in range(1, 8)],
            [model.name for model in models],
        )
        self.assertEqual(keys, [model.api_key for model in models])
        self.assertEqual({"sense-nova"}, {model.round_robin_group for model in models})

        old_models = run_label.API_MODELS
        old_cursor = run_label.API_ROUND_ROBIN_CURSOR
        run_label.API_MODELS = models
        run_label.API_ROUND_ROBIN_CURSOR = {}
        try:
            first_models = [
                run_label._api_model_iteration_order()[0].name
                for _ in range(8)
            ]
        finally:
            run_label.API_MODELS = old_models
            run_label.API_ROUND_ROBIN_CURSOR = old_cursor

        self.assertEqual(
            [
                "sense-nova-1",
                "sense-nova-2",
                "sense-nova-3",
                "sense-nova-4",
                "sense-nova-5",
                "sense-nova-6",
                "sense-nova-7",
                "sense-nova-1",
            ],
            first_models,
        )

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

    def test_transient_pool_cooldown_waits_then_retries(self):
        model = ApiModel(
            "sense-nova-1",
            "model",
            "https://example.invalid",
            "key",
            round_robin_group="sense-nova",
        )
        model.mark_failure(
            'HTTP 404: {"error":{"message":"model route not found"}}',
            cooldown=30,
        )
        old_models = run_label.API_MODELS
        run_label.API_MODELS = [model]

        def finish_cooldown(_delay):
            model.cooldown_until = 0

        try:
            with patch.object(
                run_label,
                "_call_api_fallback_once",
                side_effect=[
                    run_label.ModelCallError("temporary route outage"),
                    ("OK", 1, 1, []),
                ],
            ) as call_once:
                with patch.object(
                    run_label.time,
                    "sleep",
                    side_effect=finish_cooldown,
                ) as sleep, patch.object(run_label, "temp_log_event"):
                    result = run_label.call_api_fallback([{"role": "user", "content": "test"}])
        finally:
            run_label.API_MODELS = old_models

        self.assertEqual(("OK", 1, 1, []), result)
        self.assertEqual(2, call_once.call_count)
        sleep.assert_called_once()

    def test_non_transient_pool_failure_does_not_wait(self):
        model = ApiModel(
            "sense-nova-1",
            "model",
            "https://example.invalid",
            "key",
            round_robin_group="sense-nova",
        )
        model.mark_failure("context budget exceeded: estimated 50000 > 40000", cooldown=30)
        old_models = run_label.API_MODELS
        run_label.API_MODELS = [model]
        try:
            with patch.object(
                run_label,
                "_call_api_fallback_once",
                side_effect=run_label.ModelCallError("context budget exceeded"),
            ), patch.object(run_label.time, "sleep") as sleep:
                with self.assertRaises(run_label.ModelCallError):
                    run_label.call_api_fallback([{"role": "user", "content": "test"}])
        finally:
            run_label.API_MODELS = old_models

        sleep.assert_not_called()

    def test_missing_required_tool_moves_to_next_round_robin_key(self):
        first = ApiModel(
            "pooled-a",
            "model",
            "https://example.invalid",
            "key-a",
            round_robin_group="provider-pool",
        )
        second = ApiModel(
            "pooled-b",
            "model",
            "https://example.invalid",
            "key-b",
            round_robin_group="provider-pool",
        )
        calls = []

        def fake_chat(model, *_args, **_kwargs):
            calls.append(model.name)
            if model is first:
                return "ignored prose", 10, 20, []
            return "", 10, 20, [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "submit_result", "arguments": "{}"},
            }]

        tool = {"type": "function", "function": {"name": "submit_result", "parameters": {"type": "object"}}}
        choice = {"type": "function", "function": {"name": "submit_result"}}
        old_trace = run_label.API_CALL_TRACE
        run_label.API_CALL_TRACE = []
        try:
            with patch.object(
                run_label,
                "_api_model_iteration_order",
                return_value=[first, second],
            ), patch.object(
                run_label, "_api_chat", side_effect=fake_chat
            ), patch.object(
                run_label, "temp_log_event"
            ), patch.object(
                run_label, "API_RETRIES", 3
            ):
                _, _, _, tool_calls = run_label.call_api_fallback(
                    [{"role": "user", "content": "return structured output"}],
                    tools=[tool],
                    tool_choice=choice,
                )
        finally:
            run_label.API_CALL_TRACE = old_trace

        self.assertEqual(["pooled-a", "pooled-b"], calls)
        self.assertEqual("submit_result", tool_calls[0]["function"]["name"])

    def test_structured_caller_can_receive_prose_for_correction_round(self):
        first = ApiModel(
            "pooled-a",
            "model",
            "https://example.invalid",
            "key-a",
            round_robin_group="provider-pool",
        )
        second = ApiModel(
            "pooled-b",
            "model",
            "https://example.invalid",
            "key-b",
            round_robin_group="provider-pool",
        )
        calls = []

        def fake_chat(model, *_args, **_kwargs):
            calls.append(model.name)
            return "complete analysis without a tool call", 10, 20, []

        tool = {"type": "function", "function": {"name": "submit_result", "parameters": {"type": "object"}}}
        choice = {"type": "function", "function": {"name": "submit_result"}}
        old_trace = run_label.API_CALL_TRACE
        run_label.API_CALL_TRACE = []
        try:
            with patch.object(
                run_label,
                "_api_model_iteration_order",
                return_value=[first, second],
            ), patch.object(
                run_label, "_api_chat", side_effect=fake_chat
            ), patch.object(
                run_label, "temp_log_event"
            ), patch.object(
                run_label, "API_RETRIES", 3
            ):
                text, _, _, tool_calls = run_label.call_api_fallback(
                    [{"role": "user", "content": "return structured output"}],
                    tools=[tool],
                    tool_choice=choice,
                    accept_text_when_tool_missing=True,
                )
        finally:
            run_label.API_CALL_TRACE = old_trace

        self.assertEqual(["pooled-a"], calls)
        self.assertIn("complete analysis", text)
        self.assertEqual([], tool_calls)
        self.assertTrue(run_label.API_CALL_TRACE is old_trace)


if __name__ == "__main__":
    unittest.main()
