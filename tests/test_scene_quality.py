import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scene_quality import (  # noqa: E402
    BOUNDARY_BRIEF_TOOL,
    BOUNDARY_JURY_TOOL,
    SCENE_VERDICT_TOOL,
    SceneQualityRuntime,
    SceneSequenceAudit,
)


def _validate_speaker(value):
    value = str(value or "").strip()
    return value or None


def _same_speaker(left, right):
    return _validate_speaker(left) == _validate_speaker(right)


def _unused(*_args, **_kwargs):
    raise AssertionError("model/runtime hook must not be called by selection tests")


class SceneSelectionTests(unittest.TestCase):
    def setUp(self):
        runtime = SceneQualityRuntime(
            call_model=_unused,
            parse_tool_arguments=_unused,
            normalize_tool_call=_unused,
            log_agent=_unused,
            trace_tool_result=_unused,
            temp_log_event=_unused,
            read_novel_lines=_unused,
            search_novel=_unused,
            validate_speaker=_validate_speaker,
            is_valid_speaker=lambda value: bool(value and value != "?"),
            is_generic_speaker=lambda value: value in {"店员", "旅人", "村民"},
            same_speaker=_same_speaker,
            estimate_messages=lambda messages: sum(len(str(item)) for item in messages),
            read_tool={},
            search_tool={},
            non_person_label="非人物发声",
        )
        novel_lines = ["叙事\n"] * 40
        novel_lines[9] = "角色乙说道。\n"
        self.audit = SceneSequenceAudit(
            novel_lines,
            [(10, "第一句"), (12, "第二句")],
            runtime,
            allow_model_consensus_override=True,
        )
        self.conservative_audit = SceneSequenceAudit(
            novel_lines,
            [(10, "第一句"), (12, "第二句")],
            runtime,
        )
        segment = self.audit.segments[0]
        from scene_sequence import build_scene_packet

        self.packet = build_scene_packet(self.audit.novel_lines, self.audit.dialogue_list, segment)

    def test_final_verdict_requires_quote_type(self):
        required = SCENE_VERDICT_TOOL["function"]["parameters"]["required"]

        self.assertIn("quote_type", required)
        self.assertIn("voice_kind", required)

    @staticmethod
    def scene_card(speaker, basis="sequence_constraint", confidence="high"):
        return {
            "valid": True,
            "speaker": speaker,
            "evidence_basis": basis,
            "confidence": confidence,
            "citations": [10],
        }

    @staticmethod
    def investigation(
        speaker,
        basis="speaker_action",
        confidence="high",
        voice_kind="person",
        quote_type="direct_speech",
    ):
        return {
            "valid": True,
            "status": "resolved",
            "speaker": speaker,
            "voice_kind": voice_kind,
            "quote_type": quote_type,
            "evidence_basis": basis,
            "confidence": confidence,
            "citations": [10],
        }

    @staticmethod
    def verdict(
        speaker,
        level="direct",
        basis="speaker_action",
        confidence="high",
        status="resolved",
        voice_kind="person",
        quote_type="direct_speech",
    ):
        return {
            "valid": True,
            "status": status,
            "speaker": speaker if status == "resolved" else "",
            "voice_kind": voice_kind if status == "resolved" else "unclear",
            "quote_type": quote_type if status == "resolved" else "unclear",
            "evidence_level": level,
            "evidence_basis": basis,
            "confidence": confidence,
            "citations": [10] if status == "resolved" else [],
        }

    @staticmethod
    def contrastive(speaker, *, unanimous=True, valid=True):
        juries = [
            {
                "valid": True,
                "status": "resolved",
                "speaker": speaker,
                "confidence": "high",
                "citations": [10],
            }
            for _ in range(3)
        ]
        return {
            "valid": valid,
            "status": "resolved" if valid else "unresolved",
            "speaker": speaker if valid else "",
            "unanimous": unanimous,
            "confidence": "high",
            "citations": [10],
            "juries": juries,
            "consensus_deciders": juries if valid and unanimous else [],
            "consensus_source": "initial_juries" if valid and unanimous else "none",
        }

    def select(self, baseline, scene_cards, investigations, verdict, contrastive=None):
        return self.audit._select_decision(
            baseline,
            {"quote_type": "direct_speech", "evidence_basis": "inference", "confidence": "low"},
            scene_cards,
            investigations,
            verdict,
            self.packet,
            0,
            {},
            contrastive,
        )

    def select_conservative(self, baseline, scene_cards, investigations, verdict, contrastive=None):
        return self.conservative_audit._select_decision(
            baseline,
            {"quote_type": "direct_speech", "evidence_basis": "inference", "confidence": "low"},
            scene_cards,
            investigations,
            verdict,
            self.packet,
            0,
            {},
            contrastive,
        )

    def test_default_policy_retains_valid_baseline_against_model_consensus(self):
        scene_cards = [self.scene_card("角色乙") for _ in range(5)]
        investigations = [self.investigation("角色乙") for _ in range(5)]

        decision = self.select_conservative(
            "角色甲",
            scene_cards,
            investigations,
            self.verdict("角色乙", level="strong", basis="sequence_constraint"),
            self.contrastive("角色乙"),
        )

        self.assertEqual("角色甲", decision["speaker"])
        self.assertFalse(decision["baseline_changed"])
        self.assertFalse(decision["model_consensus_override_enabled"])
        self.assertEqual("角色乙", decision["rejected_scene_proposal"])
        self.assertEqual("角色乙", decision["rejected_boundary_proposal"])

    def test_default_policy_can_replace_an_invalid_baseline(self):
        decision = self.select_conservative(
            "?",
            [],
            [],
            self.verdict("角色乙", level="strong", basis="sequence_constraint"),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertTrue(decision["baseline_changed"])
        self.assertEqual("invalid-baseline-scene-replacement", decision["selection_source"])

    def test_default_policy_does_not_confirm_model_only_baseline_agreement(self):
        decision = self.select_conservative(
            "角色甲",
            [self.scene_card("角色甲") for _ in range(5)],
            [self.investigation("角色甲") for _ in range(5)],
            self.verdict("角色甲", level="strong", basis="sequence_constraint"),
        )

        self.assertEqual("角色甲", decision["speaker"])
        self.assertFalse(decision["confirmed_anchor"])
        self.assertEqual(
            "model-verdict-agrees-baseline-unconfirmed",
            decision["evidence_gate"],
        )

    def test_dialogue_contact_verb_creates_previous_quote_scope_constraint(self):
        packet = {
            "turns": [
                {"turn_id": "D0", "dialogue_index": 0, "line": 10},
                {"turn_id": "D1", "dialogue_index": 1, "line": 12},
            ],
            "raw_lines": [
                {"line": 10, "text": "第一句"},
                {"line": 11, "text": "角色甲向角色乙搭腔。"},
                {"line": 12, "text": "第二句"},
            ],
        }

        constraints = self.conservative_audit._format_boundary_constraints(packet, 1)

        self.assertIn("binds that preceding quote, not TARGET", constraints)

    def test_inability_to_speak_does_not_create_scope_constraint(self):
        packet = {
            "turns": [
                {"turn_id": "D0", "dialogue_index": 0, "line": 10},
                {"turn_id": "D1", "dialogue_index": 1, "line": 12},
            ],
            "raw_lines": [
                {"line": 10, "text": "第一句"},
                {"line": 11, "text": "角色甲说不出话来。"},
                {"line": 12, "text": "第二句"},
            ],
        }

        constraints = self.conservative_audit._format_boundary_constraints(packet, 1)

        self.assertIn("No deterministic adjacent speech-attribution scope", constraints)

    def test_contact_subject_detection_does_not_treat_addressee_as_speaker(self):
        self.assertEqual(
            [],
            self.conservative_audit._contact_verbs_for_speaker("向角色乙搭腔的是角色甲。", "角色乙"),
        )
        self.assertEqual(
            ["搭腔"],
            self.conservative_audit._contact_verbs_for_speaker("向角色乙搭腔的是角色甲。", "角色甲"),
        )

    def test_contact_subject_detection_handles_long_subject_action_clause(self):
        verbs = self.conservative_audit._contact_verbs_for_speaker(
            "角色甲先向旁人招呼几句后，便向角色乙搭腔。",
            "角色甲",
        )

        self.assertEqual(["搭腔"], verbs)

    def test_scope_conflict_logs_cross_role_consensus_without_overriding(self):
        packet = {
            "turns": [
                {"turn_id": "D0", "dialogue_index": 0, "line": 10},
                {"turn_id": "D1", "dialogue_index": 1, "line": 12},
            ],
            "raw_lines": [
                {"line": 10, "text": "第一句"},
                {"line": 11, "text": "角色甲向角色乙搭腔。"},
                {"line": 12, "text": "第二句"},
            ],
        }
        scene_cards = [self.scene_card("角色乙") for _ in range(2)]
        scene_cards[0]["agent"] = "SceneA"
        scene_cards[1]["agent"] = "SceneB"
        for card in scene_cards:
            card["citations"] = [12]
        investigations = [self.investigation("角色乙")]
        investigations[0]["agent"] = "InvestigatorA"
        investigations[0]["citations"] = [12]

        decision = self.conservative_audit._select_decision(
            "角色甲",
            {"quote_type": "direct_speech", "evidence_basis": "speaker_action", "confidence": "high"},
            scene_cards,
            investigations,
            self.verdict("角色甲", level="strong", basis="sequence_constraint"),
            packet,
            1,
            {
                "attribution_candidates": [
                    {"speaker": "角色甲", "verb": "搭腔", "line": 11, "text": "角色甲向角色乙搭腔。"}
                ]
            },
            baseline_reason="L11 shows 角色甲 speaking the target",
        )

        self.assertEqual("角色甲", decision["speaker"])
        self.assertFalse(decision["baseline_changed"])
        self.assertTrue(decision["baseline_scope_audit"]["conflict"])
        self.assertTrue(decision["scope_repair_consensus"]["valid"])
        self.assertEqual("角色乙", decision["scope_repair_consensus"]["speaker"])

    def test_contact_scope_conflict_requires_dedicated_repair_agents(self):
        packet = {
            "turns": [
                {"turn_id": "D0", "dialogue_index": 0, "line": 10},
                {"turn_id": "D1", "dialogue_index": 1, "line": 12},
            ],
            "raw_lines": [
                {"line": 10, "text": "第一句"},
                {"line": 11, "text": "角色甲向角色乙搭腔。"},
                {"line": 12, "text": "第二句"},
            ],
        }
        scene_cards = [self.scene_card("角色乙") for _ in range(2)]
        investigations = [self.investigation("角色乙") for _ in range(2)]
        for index, card in enumerate(scene_cards):
            card["agent"] = f"Scene{index}"
            card["citations"] = [12]
        for index, card in enumerate(investigations):
            card["agent"] = f"Investigator{index}"
            card["citations"] = [12]

        decision = self.conservative_audit._select_decision(
            "角色甲",
            {"quote_type": "direct_speech", "evidence_basis": "speaker_action", "confidence": "high"},
            scene_cards,
            investigations,
            self.verdict("角色乙", level="strong", basis="sequence_constraint"),
            packet,
            1,
            {
                "attribution_candidates": [
                    {"speaker": "角色甲", "verb": "搭腔", "line": 11, "text": "角色甲向角色乙搭腔。"}
                ]
            },
            baseline_reason="L11 shows 角色甲 speaking the target",
        )

        self.assertEqual("角色甲", decision["speaker"])
        self.assertFalse(decision["baseline_changed"])

    def test_contact_scope_conflict_can_use_dedicated_repair_consensus(self):
        packet = {
            "turns": [
                {"turn_id": "D0", "dialogue_index": 0, "line": 10},
                {"turn_id": "D1", "dialogue_index": 1, "line": 12},
            ],
            "raw_lines": [
                {"line": 10, "text": "第一句"},
                {"line": 11, "text": "角色甲向角色乙搭腔。"},
                {"line": 12, "text": "第二句"},
            ],
        }
        investigations = [self.investigation("角色乙") for _ in range(3)]
        names = [
            "TargetContactScopeRepairForward",
            "TargetContactScopeRepairReverse",
            "TargetOpenWorldInvestigator",
        ]
        for card, name in zip(investigations, names):
            card["agent"] = name
            card["citations"] = [12]

        decision = self.conservative_audit._select_decision(
            "角色甲",
            {"quote_type": "direct_speech", "evidence_basis": "speaker_action", "confidence": "high"},
            [],
            investigations,
            self.verdict("角色乙", level="strong", basis="sequence_constraint"),
            packet,
            1,
            {
                "attribution_candidates": [
                    {"speaker": "角色甲", "verb": "搭腔", "line": 11, "text": "角色甲向角色乙搭腔。"}
                ]
            },
            baseline_reason="L11 shows 角色甲 speaking the target",
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertTrue(decision["baseline_changed"])
        self.assertEqual("contact-scope-repair-override", decision["selection_source"])

    def test_scope_conflict_without_cross_role_consensus_keeps_baseline(self):
        packet = {
            "turns": [
                {"turn_id": "D0", "dialogue_index": 0, "line": 10},
                {"turn_id": "D1", "dialogue_index": 1, "line": 12},
            ],
            "raw_lines": [
                {"line": 10, "text": "第一句"},
                {"line": 11, "text": "角色甲向角色乙搭腔。"},
                {"line": 12, "text": "第二句"},
            ],
        }
        lone_card = self.scene_card("角色乙")
        lone_card["agent"] = "SceneA"
        lone_card["citations"] = [12]

        decision = self.conservative_audit._select_decision(
            "角色甲",
            {"quote_type": "direct_speech", "evidence_basis": "speaker_action", "confidence": "high"},
            [lone_card],
            [],
            self.verdict("角色甲", level="strong", basis="sequence_constraint"),
            packet,
            1,
            {
                "attribution_candidates": [
                    {"speaker": "角色甲", "verb": "搭腔", "line": 11, "text": "角色甲向角色乙搭腔。"}
                ]
            },
            baseline_reason="L11 shows 角色甲 speaking the target",
        )

        self.assertEqual("角色甲", decision["speaker"])
        self.assertFalse(decision["baseline_changed"])

    def test_boundary_tools_require_explicit_weakness_and_relations(self):
        brief_required = BOUNDARY_BRIEF_TOOL["function"]["parameters"]["required"]
        jury_required = BOUNDARY_JURY_TOOL["function"]["parameters"]["required"]

        self.assertIn("weakest_assumption", brief_required)
        self.assertIn("target_to_previous", jury_required)
        self.assertIn("target_to_next", jury_required)

    def test_unanimous_mirrored_boundary_jury_can_override_correlated_majority(self):
        decision = self.select(
            "角色甲",
            [self.scene_card("角色甲") for _ in range(4)] + [self.scene_card("角色乙")],
            [self.investigation("角色甲") for _ in range(3)],
            self.verdict(
                "角色甲",
                level="corroborated_sequence",
                basis="sequence_constraint",
                confidence="medium",
            ),
            self.contrastive("角色乙"),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("unanimous-mirrored-boundary-jury", decision["evidence_gate"])

    def test_boundary_jury_disagreement_cannot_override(self):
        contrastive = self.contrastive("角色乙", unanimous=False, valid=False)
        contrastive["juries"][2]["speaker"] = "角色甲"
        decision = self.select(
            "角色甲",
            [],
            [],
            self.verdict("角色甲", level="weak", basis="inference"),
            contrastive,
        )

        self.assertEqual("角色甲", decision["speaker"])

    def test_disagreement_referees_require_independent_scene_support(self):
        contrastive = self.contrastive("角色乙")
        contrastive["consensus_deciders"] = contrastive["juries"][:2]
        contrastive["consensus_source"] = "disagreement_referees"
        weak = self.select(
            "角色甲",
            [self.scene_card("角色乙")],
            [],
            self.verdict("角色甲", level="weak", basis="inference"),
            contrastive,
        )
        supported = self.select(
            "角色甲",
            [self.scene_card("角色乙") for _ in range(3)],
            [],
            self.verdict("角色甲", level="weak", basis="inference"),
            contrastive,
        )

        self.assertEqual("角色甲", weak["speaker"])
        self.assertEqual("角色乙", supported["speaker"])
        self.assertEqual("mirrored-referees-plus-scene-chain", supported["evidence_gate"])

    def test_generic_role_and_named_identity_skip_boundary_debate(self):
        report, pec, ec = self.audit._run_contrastive_boundary(
            "店员",
            [self.scene_card("店员") for _ in range(3)],
            [
                self.investigation("角色乙", basis="identity_alias"),
                self.investigation("角色乙", basis="identity_alias"),
            ],
            self.verdict("店员", level="weak", basis="inference"),
            self.packet,
            0,
            "packet",
            {},
        )

        self.assertEqual("not_run", report["status"])
        self.assertIn("identity canonicalization", report["reason"])
        self.assertEqual((0, 0), (pec, ec))

    def test_neighboring_scene_speaker_can_seed_missing_contrastive_candidate(self):
        candidates = self.audit._contrastive_candidate_pair(
            "角色甲",
            [self.scene_card("角色甲")],
            [self.investigation("角色甲")],
            self.verdict("角色甲", level="weak", basis="inference"),
            ["角色乙"],
        )

        self.assertEqual(["角色甲", "角色乙"], [item["speaker"] for item in candidates])

    def test_neighboring_scene_candidates_use_packet_id_field(self):
        scene_solution = {
            "packet": {
                "turns": [
                    {"id": "D0", "dialogue_index": 0, "line": 10},
                    {"id": "D1", "dialogue_index": 1, "line": 11},
                ]
            },
            "maps": [
                {
                    "agent": "SceneA",
                    "valid": True,
                    "assignments": {
                        "D0": {"speaker": "角色乙"},
                        "D1": {"speaker": "角色甲"},
                    },
                }
            ],
        }

        candidates = self.conservative_audit._neighboring_scene_candidates(
            scene_solution,
            1,
            "角色甲",
        )

        self.assertEqual(["角色乙"], candidates)

    def test_boundary_jury_cannot_overrule_a_direct_anchor(self):
        decision = self.select(
            "角色甲",
            [],
            [self.investigation("角色乙", basis="explicit_attribution")],
            self.verdict("角色乙", level="direct", basis="explicit_attribution"),
            self.contrastive("角色甲"),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("direct-target-binding", decision["evidence_gate"])

    def test_scene_majority_cannot_override_a_weak_verdict(self):
        decision = self.select(
            "角色甲",
            [self.scene_card("角色乙") for _ in range(3)],
            [],
            self.verdict("角色乙", level="weak", basis="inference"),
        )

        self.assertEqual("角色甲", decision["speaker"])
        self.assertFalse(decision["baseline_changed"])

    def test_direct_binding_can_override_with_independent_confirmation(self):
        decision = self.select(
            "角色甲",
            [],
            [self.investigation("角色乙", basis="explicit_attribution")],
            self.verdict("角色乙", level="direct", basis="explicit_attribution"),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("direct-target-binding", decision["evidence_gate"])

    def test_previous_quote_post_attribution_cannot_bind_next_quote(self):
        packet = {
            "raw_lines": [
                {"line": 9, "text": "「上一句。」"},
                {"line": 10, "text": "角色乙回答。"},
                {"line": 11, "text": "「下一句。」"},
                {"line": 12, "text": "角色甲说道。"},
            ]
        }

        self.assertFalse(
            self.audit._has_direct_speech_binding("角色乙", [10, 11], packet, 11)
        )
        self.assertTrue(
            self.audit._has_direct_speech_binding("角色甲", [12], packet, 11)
        )

    def test_boundary_constraints_consume_previous_post_attribution(self):
        packet = {
            "turns": [
                {"turn_id": "D0", "dialogue_index": 0, "line": 9},
                {"turn_id": "D1", "dialogue_index": 1, "line": 12},
                {"turn_id": "D2", "dialogue_index": 2, "line": 15},
            ],
            "raw_lines": [
                {"line": 9, "text": "「上一句。」"},
                {"line": 10, "text": "角色甲说道。"},
                {"line": 11, "text": "听者沉默。"},
                {"line": 12, "text": "「目标句。」"},
                {"line": 13, "text": "听者抬头。"},
                {"line": 14, "text": "角色乙接着说道："},
                {"line": 15, "text": "「下一句。」"},
            ],
        }

        constraints = self.audit._format_boundary_constraints(packet, 1)

        self.assertIn("L10", constraints)
        self.assertIn("binds that preceding quote, not TARGET", constraints)
        self.assertIn("L14", constraints)
        self.assertIn("introduces that next quote", constraints)

    def test_preceding_attribution_with_colon_can_bind_next_quote(self):
        packet = {
            "raw_lines": [
                {"line": 9, "text": "「上一句。」"},
                {"line": 10, "text": "角色乙接着说道："},
                {"line": 11, "text": "「下一句。」"},
            ]
        }

        self.assertTrue(
            self.audit._has_direct_speech_binding("角色乙", [10], packet, 11)
        )

    def test_named_expression_conveying_quote_restores_person(self):
        raw_line = next(item for item in self.packet["raw_lines"] if item["line"] == 10)
        original = raw_line["text"]
        raw_line["text"] = "角色乙看了一眼，仿佛在说「第一句」似地笑着。"
        try:
            decision = self.select(
                "非人物发声",
                [self.scene_card("角色乙"), self.scene_card("角色乙")],
                [
                    self.investigation(
                        "非人物发声",
                        basis="quote_type",
                        voice_kind="non_person",
                        quote_type="thought_or_narration",
                    )
                ],
                self.verdict(
                    "非人物发声",
                    level="quote_type",
                    basis="quote_type",
                    voice_kind="non_person",
                    quote_type="thought_or_narration",
                ),
            )
        finally:
            raw_line["text"] = original

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("person", decision["voice_kind"])
        self.assertEqual("embedded_quote", decision["quote_type"])
        self.assertEqual("conveyed-quote-person-restoration", decision["selection_source"])

    def test_named_demeanor_container_conveying_quote_restores_person(self):
        raw_line = next(item for item in self.packet["raw_lines"] if item["line"] == 10)
        original = raw_line["text"]
        raw_line["text"] = "角色乙一副「第一句」的模样转过身去。"
        try:
            decision = self.select(
                "非人物发声",
                [self.scene_card("角色乙") for _ in range(3)],
                [self.investigation("角色乙", basis="speaker_action", quote_type="embedded_quote")],
                self.verdict(
                    "角色乙",
                    level="direct",
                    basis="speaker_action",
                    quote_type="embedded_quote",
                ),
            )
        finally:
            raw_line["text"] = original

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("conveyed-quote-person-restoration", decision["selection_source"])

    def test_adjacent_action_cannot_masquerade_as_direct_attribution(self):
        original = next(item for item in self.packet["raw_lines"] if item["line"] == 10)["text"]
        next(item for item in self.packet["raw_lines"] if item["line"] == 10)["text"] = "角色乙摇了摇头。"
        try:
            decision = self.select(
                "角色甲",
                [self.scene_card("角色乙") for _ in range(3)],
                [self.investigation("角色乙", basis="explicit_attribution")],
                self.verdict("角色乙", level="direct", basis="explicit_attribution"),
            )
        finally:
            next(item for item in self.packet["raw_lines"] if item["line"] == 10)["text"] = original

        self.assertEqual("角色甲", decision["speaker"])
        self.assertEqual("insufficient-to-override", decision["evidence_gate"])

    def test_adjacent_action_ambiguity_blocks_a_pure_sequence_override(self):
        raw_line = next(item for item in self.packet["raw_lines"] if item["line"] == 10)
        original = raw_line["text"]
        raw_line["text"] = "角色甲说得轻松，角色乙摇了摇头。"
        try:
            decision = self.select(
                "角色甲",
                [self.scene_card("角色乙") for _ in range(3)],
                [
                    self.investigation("角色乙", basis="sequence_constraint"),
                    self.investigation("角色乙", basis="sequence_constraint"),
                ],
                self.verdict(
                    "角色乙",
                    level="corroborated_sequence",
                    basis="sequence_constraint",
                ),
            )
        finally:
            raw_line["text"] = original

        self.assertEqual("角色甲", decision["speaker"])
        self.assertTrue(decision["adjacent_non_speech_ambiguity"])

    def test_unspoken_container_can_override_a_person_baseline(self):
        raw_line = next(item for item in self.packet["raw_lines"] if item["line"] == 10)
        original = raw_line["text"]
        raw_line["text"] = "「一句反驳。」角色乙省略掉这句反驳话语。"
        try:
            report = self.investigation(
                "非人物发声",
                basis="quote_type",
                voice_kind="non_person",
                quote_type="thought_or_narration",
            )
            decision = self.select(
                "角色乙",
                [self.scene_card("角色乙", basis="quote_type")],
                [report],
                self.verdict(
                    "非人物发声",
                    level="quote_type",
                    basis="quote_type",
                    voice_kind="non_person",
                    quote_type="thought_or_narration",
                ),
            )
        finally:
            raw_line["text"] = original

        self.assertEqual("非人物发声", decision["speaker"])
        self.assertEqual("deterministic-unspoken-container", decision["evidence_gate"])

    def test_partially_spoken_quote_is_not_an_unspoken_container(self):
        raw_line = next(item for item in self.packet["raw_lines"] if item["line"] == 10)
        original = raw_line["text"]
        raw_line["text"] = "角色乙原本要说：「一句话。」但说到一半被打断。"
        try:
            detected = self.audit._has_nonperson_quote_container(10, self.packet)
        finally:
            raw_line["text"] = original

        self.assertFalse(detected)

    def test_started_speech_restores_person_against_nonperson_verdict(self):
        raw_line = next(item for item in self.packet["raw_lines"] if item["line"] == 10)
        original = raw_line["text"]
        raw_line["text"] = "角色乙原本要说：「一句话。」但说到一半被打断。"
        try:
            decision = self.select(
                "非人物发声",
                [self.scene_card("角色乙", basis="explicit_attribution") for _ in range(3)],
                [
                    self.investigation(
                        "非人物发声",
                        basis="quote_type",
                        voice_kind="non_person",
                        quote_type="thought_or_narration",
                    )
                ],
                self.verdict(
                    "非人物发声",
                    level="quote_type",
                    basis="quote_type",
                    voice_kind="non_person",
                    quote_type="thought_or_narration",
                ),
            )
        finally:
            raw_line["text"] = original

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("started-speech-plus-scene-consensus", decision["evidence_gate"])

    def test_named_person_vocalization_restores_person_label(self):
        raw_line = next(item for item in self.packet["raw_lines"] if item["line"] == 10)
        original = raw_line["text"]
        raw_line["text"] = "角色乙熟睡着，还「呼噜」地发出毫无警觉性的鼾声。"
        try:
            decision = self.select(
                "非人物发声",
                [self.scene_card("角色乙", basis="quote_type") for _ in range(3)],
                [
                    self.investigation(
                        "非人物发声",
                        basis="quote_type",
                        voice_kind="non_person",
                        quote_type="sound_or_text",
                    )
                ],
                self.verdict(
                    "非人物发声",
                    level="quote_type",
                    basis="quote_type",
                    voice_kind="non_person",
                    quote_type="sound_or_text",
                ),
            )
        finally:
            raw_line["text"] = original

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("bound-person-vocalization-plus-scene-consensus", decision["evidence_gate"])

    def test_verified_identity_can_replace_a_generic_role(self):
        decision = self.select(
            "店员",
            [],
            [self.investigation("角色乙", basis="identity_alias")],
            self.verdict("角色乙", level="identity", basis="identity_alias"),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("verified-identity-chain", decision["evidence_gate"])

    def test_sequence_override_requires_two_maps_and_two_investigations(self):
        verdict = self.verdict(
            "角色乙",
            level="corroborated_sequence",
            basis="sequence_constraint",
        )
        one_investigation = self.select(
            "角色甲",
            [self.scene_card("角色乙"), self.scene_card("角色乙")],
            [self.investigation("角色乙", basis="sequence_constraint")],
            verdict,
        )
        two_investigations = self.select(
            "角色甲",
            [self.scene_card("角色乙"), self.scene_card("角色乙")],
            [
                self.investigation("角色乙", basis="sequence_constraint"),
                self.investigation("角色乙", basis="speaker_action"),
            ],
            verdict,
        )

        self.assertEqual("角色甲", one_investigation["speaker"])
        self.assertEqual("角色乙", two_investigations["speaker"])

    def test_unanimous_scene_plus_one_independent_investigation_can_override(self):
        decision = self.select(
            "角色甲",
            [self.scene_card("角色乙") for _ in range(3)],
            [self.investigation("角色乙", basis="sequence_constraint")],
            self.verdict(
                "角色乙",
                level="direct",
                basis="speaker_action",
            ),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual(
            "unanimous-scene-plus-independent-investigation",
            decision["evidence_gate"],
        )

    def test_three_independent_investigations_can_recover_missing_scene_candidate(self):
        decision = self.select(
            "角色甲",
            [self.scene_card("角色丙") for _ in range(3)],
            [
                self.investigation("角色乙", basis="speaker_action"),
                self.investigation("角色乙", basis="sequence_constraint"),
                self.investigation("角色乙", basis="quote_type"),
            ],
            self.verdict(
                "角色乙",
                level="corroborated_sequence",
                basis="sequence_constraint",
            ),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("three-independent-investigations", decision["evidence_gate"])

    def test_scene_chain_can_override_an_unresolved_judge(self):
        decision = self.select(
            "角色甲",
            [self.scene_card("角色乙") for _ in range(4)],
            [self.investigation("角色乙", basis="sequence_constraint")],
            self.verdict(
                "",
                level="insufficient",
                basis="unknown",
                confidence="low",
                status="unresolved",
            ),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("scene-chain-conflict-override", decision["selection_source"])
        self.assertEqual("four-scene-maps-plus-investigation", decision["evidence_gate"])

    def test_four_of_five_scene_maps_can_override_random_baseline(self):
        decision = self.select(
            "角色甲",
            [self.scene_card("角色乙") for _ in range(4)] + [self.scene_card("角色丙")],
            [],
            self.verdict(
                "",
                level="insufficient",
                basis="unknown",
                confidence="low",
                status="unresolved",
            ),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("four-of-five-scene-map-supermajority", decision["evidence_gate"])

    def test_three_of_five_with_reverse_decoder_can_override_random_baseline(self):
        supporting = [self.scene_card("角色乙") for _ in range(3)]
        supporting[-1]["agent"] = "SceneReverseMapper"
        decision = self.select(
            "角色甲",
            supporting + [self.scene_card("角色丙"), self.scene_card("角色丙")],
            [],
            self.verdict(
                "",
                level="insufficient",
                basis="unknown",
                confidence="low",
                status="unresolved",
            ),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("three-of-five-scene-decoder-majority", decision["evidence_gate"])

    def test_three_of_five_without_specialized_decoder_retains_baseline(self):
        decision = self.select(
            "角色甲",
            [self.scene_card("角色乙") for _ in range(3)]
            + [self.scene_card("角色丙"), self.scene_card("角色丙")],
            [],
            self.verdict(
                "",
                level="insufficient",
                basis="unknown",
                confidence="low",
                status="unresolved",
            ),
        )

        self.assertEqual("角色甲", decision["speaker"])

    def test_three_scene_maps_plus_blind_investigation_can_override(self):
        blind_report = self.investigation("角色乙", basis="sequence_constraint")
        blind_report["agent"] = "TargetOpenWorldInvestigator"
        decision = self.select(
            "角色甲",
            [self.scene_card("角色乙") for _ in range(3)],
            [blind_report],
            self.verdict(
                "",
                level="insufficient",
                basis="unknown",
                confidence="low",
                status="unresolved",
            ),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("three-scene-maps-plus-blind-investigation", decision["evidence_gate"])

    def test_unanimous_generic_scene_role_can_be_canonicalized_by_two_identity_reports(self):
        decision = self.select(
            "店员",
            [self.scene_card("店员") for _ in range(5)],
            [
                self.investigation("角色乙", basis="identity_alias"),
                self.investigation("角色乙", basis="explicit_attribution"),
            ],
            self.verdict("店员", level="corroborated_sequence", basis="sequence_constraint"),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("scene-role-identity-canonicalization", decision["selection_source"])
        self.assertEqual(
            "unanimous-scene-role-plus-two-identity-investigations",
            decision["evidence_gate"],
        )

    def test_scene_chain_does_not_override_adjacent_action_ambiguity(self):
        raw_line = next(item for item in self.packet["raw_lines"] if item["line"] == 10)
        original = raw_line["text"]
        raw_line["text"] = "角色乙摇了摇头。"
        try:
            decision = self.select(
                "角色甲",
                [self.scene_card("角色乙") for _ in range(5)],
                [self.investigation("角色乙", basis="sequence_constraint") for _ in range(2)],
                self.verdict(
                    "",
                    level="insufficient",
                    basis="unknown",
                    confidence="low",
                    status="unresolved",
                ),
            )
        finally:
            raw_line["text"] = original

        self.assertEqual("角色甲", decision["speaker"])

    def test_unresolved_verdict_retains_valid_baseline(self):
        decision = self.select(
            "角色甲",
            [self.scene_card("角色乙")],
            [],
            self.verdict("", status="unresolved"),
        )

        self.assertEqual("角色甲", decision["speaker"])
        self.assertEqual("scene-evidence-baseline-retained", decision["selection_source"])

    def test_matching_verdict_is_an_anchor_only_when_level_and_basis_agree(self):
        inconsistent = self.select(
            "角色甲",
            [],
            [],
            self.verdict("角色甲", level="direct", basis="sequence_constraint"),
        )
        direct = self.select(
            "角色乙",
            [],
            [],
            self.verdict("角色乙", level="direct", basis="explicit_attribution"),
        )

        self.assertFalse(inconsistent["confirmed_anchor"])
        self.assertTrue(direct["confirmed_anchor"])

    def test_resolved_verdict_replaces_an_invalid_baseline(self):
        decision = self.select(
            "?",
            [],
            [],
            self.verdict("角色乙", level="direct", basis="speaker_action"),
        )

        self.assertEqual("角色乙", decision["speaker"])
        self.assertEqual("invalid-baseline-scene-replacement", decision["selection_source"])


class MessageCompactionTests(unittest.TestCase):
    def test_compaction_never_leaves_an_orphan_tool_message(self):
        runtime = SceneQualityRuntime(
            call_model=_unused,
            parse_tool_arguments=_unused,
            normalize_tool_call=_unused,
            log_agent=_unused,
            trace_tool_result=_unused,
            temp_log_event=_unused,
            read_novel_lines=_unused,
            search_novel=_unused,
            validate_speaker=_validate_speaker,
            is_valid_speaker=lambda value: bool(value),
            is_generic_speaker=lambda _value: False,
            same_speaker=_same_speaker,
            estimate_messages=lambda messages: sum(len(str(item)) for item in messages),
            read_tool={},
            search_tool={},
            non_person_label="非人物发声",
            context_limit=1200,
            max_output_tokens=100,
        )
        audit = SceneSequenceAudit(["叙事\n"] * 20, [(5, "一句")], runtime)
        messages = [
            {"role": "system", "content": "s" * 300},
            {"role": "user", "content": "u" * 300},
            {
                "role": "assistant",
                "content": "a" * 300,
                "tool_calls": [{"id": "call-1", "function": {"name": "read", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "t" * 500},
            {"role": "user", "content": "continue"},
        ]

        compacted = audit._compact_messages(messages)

        for index, message in enumerate(compacted):
            if message["role"] == "tool":
                self.assertGreater(index, 0)
                self.assertEqual("assistant", compacted[index - 1]["role"])
                self.assertTrue(compacted[index - 1].get("tool_calls"))


class RequiredToolFailureTests(unittest.TestCase):
    def test_exhausted_model_pool_returns_invalid_result_instead_of_raising(self):
        failures = []

        def fail_model(*_args, **_kwargs):
            failures.append(1)
            raise RuntimeError("all providers returned no tool call")

        runtime = SceneQualityRuntime(
            call_model=fail_model,
            parse_tool_arguments=_unused,
            normalize_tool_call=_unused,
            log_agent=lambda *_args, **_kwargs: None,
            trace_tool_result=_unused,
            temp_log_event=lambda *_args, **_kwargs: None,
            read_novel_lines=_unused,
            search_novel=_unused,
            validate_speaker=_validate_speaker,
            is_valid_speaker=lambda value: bool(value),
            is_generic_speaker=lambda _value: False,
            same_speaker=_same_speaker,
            estimate_messages=lambda _messages: 0,
            read_tool={},
            search_tool={},
            non_person_label="非人物发声",
            retries=2,
        )
        audit = SceneSequenceAudit(["叙事\n"] * 20, [(5, "一句")], runtime)

        _, normalized, _, _ = audit._run_required_tool(
            "FixtureAgent",
            "fixture",
            "system",
            "user",
            SCENE_VERDICT_TOOL,
            "submit_scene_verdict",
            lambda _args: (True, {}),
            {"agents": {}},
        )

        self.assertEqual(1, len(failures))
        self.assertFalse(normalized["valid"])
        self.assertIn("model call failed", normalized["errors"][0])

    def test_investigator_pool_failure_returns_unresolved(self):
        failures = []

        def fail_model(*_args, **_kwargs):
            failures.append(1)
            raise RuntimeError("provider pool unavailable")

        runtime = SceneQualityRuntime(
            call_model=fail_model,
            parse_tool_arguments=_unused,
            normalize_tool_call=_unused,
            log_agent=lambda *_args, **_kwargs: None,
            trace_tool_result=_unused,
            temp_log_event=lambda *_args, **_kwargs: None,
            read_novel_lines=_unused,
            search_novel=_unused,
            validate_speaker=_validate_speaker,
            is_valid_speaker=lambda value: bool(value),
            is_generic_speaker=lambda _value: False,
            same_speaker=_same_speaker,
            estimate_messages=lambda _messages: 0,
            read_tool={},
            search_tool={},
            non_person_label="非人物发声",
            deep_tool_rounds=2,
        )
        audit = SceneSequenceAudit(["叙事\n"] * 20, [(5, "一句")], runtime)
        from scene_sequence import build_scene_packet

        packet = build_scene_packet(audit.novel_lines, audit.dialogue_list, audit.segments[0])
        report, _, _ = audit._run_investigator(
            "FixtureInvestigator",
            "investigate",
            "target packet",
            [],
            [],
            packet,
            {"agents": {}},
        )

        self.assertEqual(3, len(failures))
        self.assertEqual("unresolved", report["status"])
        self.assertIn("final model call failed", report["reason"])

    def test_investigator_forces_structured_submission_after_empty_tool_round(self):
        calls = []
        submitted = {
            "status": "resolved",
            "speaker": "角色甲",
            "voice_kind": "person",
            "quote_type": "direct_speech",
            "evidence_basis": "explicit_attribution",
            "confidence": "high",
            "citations": [5],
            "reason": "L5 explicitly binds the quote to 角色甲 with a speech verb.",
            "rejected_candidates": [],
        }

        def call_model(_messages, *, tools, label, request_timeout, tool_choice):
            calls.append({"tools": tools, "label": label, "tool_choice": tool_choice})
            if len(calls) == 1:
                return "analysis without a tool", 10, 20, []
            return (
                "",
                11,
                21,
                [{"function": {"name": "submit_target_evidence", "arguments": submitted}}],
            )

        runtime = SceneQualityRuntime(
            call_model=call_model,
            parse_tool_arguments=lambda _name, arguments: (arguments, ""),
            normalize_tool_call=_unused,
            log_agent=lambda *_args, **_kwargs: None,
            trace_tool_result=_unused,
            temp_log_event=lambda *_args, **_kwargs: None,
            read_novel_lines=_unused,
            search_novel=_unused,
            validate_speaker=_validate_speaker,
            is_valid_speaker=lambda value: bool(value),
            is_generic_speaker=lambda _value: False,
            same_speaker=_same_speaker,
            estimate_messages=lambda _messages: 0,
            read_tool={"type": "function", "function": {"name": "read_novel_lines"}},
            search_tool={"type": "function", "function": {"name": "search_novel"}},
            non_person_label="非人物发声",
            deep_tool_rounds=8,
        )
        audit = SceneSequenceAudit(["叙事\n"] * 20, [(5, "一句")], runtime)
        from scene_sequence import build_scene_packet

        packet = build_scene_packet(audit.novel_lines, audit.dialogue_list, audit.segments[0])
        report, pec, ec = audit._run_investigator(
            "FixtureInvestigator",
            "investigate",
            "target packet",
            [],
            [],
            packet,
            {"agents": {}},
        )

        self.assertEqual("resolved", report["status"])
        self.assertEqual("角色甲", report["speaker"])
        self.assertEqual((21, 41), (pec, ec))
        self.assertEqual(2, len(calls))
        self.assertEqual("auto", calls[0]["tool_choice"])
        self.assertEqual(
            {"type": "function", "function": {"name": "submit_target_evidence"}},
            calls[1]["tool_choice"],
        )
        self.assertEqual("submit_target_evidence", calls[1]["tools"][0]["function"]["name"])

    def test_investigator_does_not_repeat_exhausted_forced_submission(self):
        calls = []

        def call_model(_messages, *, tools, label, request_timeout, tool_choice):
            calls.append({"tools": tools, "label": label, "tool_choice": tool_choice})
            if len(calls) == 1:
                return "analysis without a tool", 10, 20, []
            raise RuntimeError("all providers rejected the required tool")

        runtime = SceneQualityRuntime(
            call_model=call_model,
            parse_tool_arguments=_unused,
            normalize_tool_call=_unused,
            log_agent=lambda *_args, **_kwargs: None,
            trace_tool_result=_unused,
            temp_log_event=lambda *_args, **_kwargs: None,
            read_novel_lines=_unused,
            search_novel=_unused,
            validate_speaker=_validate_speaker,
            is_valid_speaker=lambda value: bool(value),
            is_generic_speaker=lambda _value: False,
            same_speaker=_same_speaker,
            estimate_messages=lambda _messages: 0,
            read_tool={"type": "function", "function": {"name": "read_novel_lines"}},
            search_tool={"type": "function", "function": {"name": "search_novel"}},
            non_person_label="非人物发声",
            deep_tool_rounds=8,
        )
        audit = SceneSequenceAudit(["叙事\n"] * 20, [(5, "一句")], runtime)
        from scene_sequence import build_scene_packet

        packet = build_scene_packet(audit.novel_lines, audit.dialogue_list, audit.segments[0])
        report, pec, ec = audit._run_investigator(
            "FixtureInvestigator",
            "investigate",
            "target packet",
            [],
            [],
            packet,
            {"agents": {}},
        )

        self.assertEqual("unresolved", report["status"])
        self.assertIn("provider-pool exhaustion", report["reason"])
        self.assertEqual((10, 20), (pec, ec))
        self.assertEqual(2, len(calls))
        self.assertEqual(
            {"type": "function", "function": {"name": "submit_target_evidence"}},
            calls[1]["tool_choice"],
        )


class SceneCacheTests(unittest.TestCase):
    def test_scene_maps_are_solved_once_and_later_targets_reference_the_cache(self):
        runtime = SceneQualityRuntime(
            call_model=_unused,
            parse_tool_arguments=_unused,
            normalize_tool_call=_unused,
            log_agent=_unused,
            trace_tool_result=_unused,
            temp_log_event=_unused,
            read_novel_lines=_unused,
            search_novel=_unused,
            validate_speaker=_validate_speaker,
            is_valid_speaker=lambda value: bool(value and value != "?"),
            is_generic_speaker=lambda _value: False,
            same_speaker=_same_speaker,
            estimate_messages=lambda messages: sum(len(str(item)) for item in messages),
            read_tool={},
            search_tool={},
            non_person_label="非人物发声",
        )
        audit = SceneSequenceAudit(
            ["叙事\n"] * 30,
            [(10, "第一句"), (12, "第二句")],
            runtime,
        )
        solve_calls = []

        def solve_scene(packet, _round_log):
            solve_calls.append(packet["scene_id"])
            assignments = {
                turn["id"]: {
                    "turn_id": turn["id"],
                    "line": turn["line"],
                    "speaker": "角色甲",
                    "quote_type": "direct_speech",
                    "evidence_basis": "inference",
                    "confidence": "low",
                    "citations": [turn["line"]],
                    "reason": "fixture",
                }
                for turn in packet["turns"]
            }
            return {"packet": packet, "maps": [{"valid": True, "assignments": assignments}]}, 0, 0

        audit._solve_scene = solve_scene
        audit._run_investigator = lambda *_args, **_kwargs: (audit._unresolved_card("fixture"), 0, 0)
        audit._run_verdict = lambda *_args, **_kwargs: (audit._unresolved_verdict("fixture"), 0, 0)

        first = audit.decide(0, 10, "第一句", "角色甲", {}, "baseline", {}, {"trace_id": "first"})[-1]
        second = audit.decide(1, 12, "第二句", "角色甲", {}, "baseline", {}, {"trace_id": "second"})[-1]

        self.assertEqual(1, len(solve_calls))
        self.assertFalse(first["scene"]["cache_hit"])
        self.assertTrue(second["scene"]["cache_hit"])
        self.assertTrue(first["scene_maps"])
        self.assertEqual([], second["scene_maps"])
        self.assertEqual("first", second["scene_map_reference"]["source_trace_id"])

    def test_identical_scene_sequences_are_hidden_as_one_hypothesis(self):
        runtime = SceneQualityRuntime(
            call_model=_unused,
            parse_tool_arguments=_unused,
            normalize_tool_call=_unused,
            log_agent=_unused,
            trace_tool_result=_unused,
            temp_log_event=_unused,
            read_novel_lines=_unused,
            search_novel=_unused,
            validate_speaker=_validate_speaker,
            is_valid_speaker=lambda value: bool(value),
            is_generic_speaker=lambda _value: False,
            same_speaker=_same_speaker,
            estimate_messages=lambda messages: sum(len(str(item)) for item in messages),
            read_tool={},
            search_tool={},
            non_person_label="非人物发声",
        )
        audit = SceneSequenceAudit(["叙事\n"] * 30, [(10, "第一句"), (12, "第二句")], runtime)
        from scene_sequence import build_scene_packet

        packet = build_scene_packet(audit.novel_lines, audit.dialogue_list, audit.segments[0])

        def assignments(speakers):
            return {
                turn["id"]: {
                    "speaker": speaker,
                    "evidence_basis": "inference",
                    "confidence": "low",
                    "citations": [turn["line"]],
                    "reason": "fixture",
                }
                for turn, speaker in zip(packet["turns"], speakers)
            }

        solution = {
            "packet": packet,
            "maps": [
                {"valid": True, "assignments": assignments(["角色甲", "角色乙"])},
                {"valid": True, "assignments": assignments(["角色甲", "角色乙"])},
                {"valid": True, "assignments": assignments(["角色乙", "角色甲"])},
            ],
        }

        sequences = audit._neutralize_scene_sequences(solution, 10)

        self.assertEqual(2, len(sequences))
        rendered = audit._format_neutral_sequences(sequences)
        self.assertIn("no source identities or support counts", rendered)


if __name__ == "__main__":
    unittest.main()
