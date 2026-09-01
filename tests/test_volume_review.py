import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from volume_review import (  # noqa: E402
    JURY_ORIENTATIONS,
    JURY_SPECS,
    VERDICT_TOOL,
    VolumeReviewRuntime,
    VolumeSecondPassReviewer,
    _structured_payloads_from_text,
    build_review_units,
    build_scene_review_units,
    derive_deterministic_sequence_constraints,
    format_review_packet,
)
from quote_occurrence import align_quote_occurrences  # noqa: E402


def valid_speaker(value):
    value = (value or "").strip()
    return value if value and len(value) <= 15 else None


class FakeModel:
    def __init__(self, speaker_by_id):
        self.speaker_by_id = dict(speaker_by_id)
        self.calls = 0

    def __call__(self, messages, *, tools, label, **kwargs):
        self.calls += 1
        expected_name = tools[0]["function"]["name"]
        user_text = "\n".join(
            message["content"] for message in messages if message.get("role") == "user"
        )
        if expected_name == "submit_volume_assignments":
            focus_line = re.search(r"Focus ids: ([^\n]+)", user_text).group(1)
            ids = [item.strip() for item in focus_line.split(",")]
            args = {
                "assignments": [
                    {
                        "id": dialogue_id,
                        "speaker": self.speaker_by_id[dialogue_id],
                        "quote_type": "direct_speech",
                        "evidence_basis": "explicit_attribution",
                        "confidence": "high",
                        "citations": [1],
                        "reason": "The raw line explicitly binds the quote.",
                    }
                    for dialogue_id in ids
                ]
            }
        else:
            dialogue_id = re.search(
                r"\[(?:Single target|Single recovery target)\]\n(D\d+)(?:;| only)",
                user_text,
            ).group(1)
            args = {
                "id": dialogue_id,
                "speaker": self.speaker_by_id[dialogue_id],
                "quote_type": "direct_speech",
                "evidence_basis": "explicit_attribution",
                "confidence": "high",
                "citations": [1],
                "reason": "The explicit attribution survives both reconstructions.",
            }
        tool_call = {
            "function": {
                "name": expected_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            }
        }
        return "", 10, 5, [tool_call]


class AlwaysFailModel(FakeModel):
    def __call__(self, messages, *, tools, label, **kwargs):
        self.calls += 1
        raise RuntimeError("simulated permanent malformed response")


class InterruptingModel(FakeModel):
    def __call__(self, messages, *, tools, label, **kwargs):
        self.calls += 1
        raise KeyboardInterrupt("simulated user interruption")


class ProseThenToolModel(FakeModel):
    def __call__(self, messages, *, tools, label, **kwargs):
        expected_name = tools[0]["function"]["name"] if tools else ""
        if expected_name == "submit_volume_assignments":
            self.calls += 1
            return "I completed the analysis but forgot the required tool.", 10, 5, []
        return super().__call__(messages, tools=tools, label=label, **kwargs)


class ProseVerdictThenCompactModel(FakeModel):
    def __call__(self, messages, *, tools, label, **kwargs):
        if "-Compact-A" not in label:
            self.calls += 1
            return "I am still analyzing and did not submit the verdict.", 10, 5, []
        return super().__call__(messages, tools=tools, label=label, **kwargs)


class ProseVerdictWithJsonCompilerModel(FakeModel):
    def __call__(self, messages, *, tools, label, **kwargs):
        self.calls += 1
        if "-CompactSerializer-A" not in label:
            return "I identified the speaker but omitted the structured verdict.", 10, 5, []
        user_text = "\n".join(
            message["content"] for message in messages if message.get("role") == "user"
        )
        dialogue_id = re.search(r"\[Target id\]\n(D\d+)", user_text).group(1)
        payload = {
            "id": dialogue_id,
            "speaker": self.speaker_by_id[dialogue_id],
            "quote_type": "direct_speech",
            "evidence_basis": "explicit_attribution",
            "confidence": "high",
            "citations": [1],
            "reason": "The raw attribution identifies the speaker.",
        }
        return json.dumps(payload, ensure_ascii=False), 10, 5, []


class FailOnSecondRecoveryTargetModel(FakeModel):
    def __call__(self, messages, *, tools, label, **kwargs):
        user_text = "\n".join(
            message["content"] for message in messages if message.get("role") == "user"
        )
        if "D1 only" in user_text:
            self.calls += 1
            raise RuntimeError("simulated second-item interruption")
        return super().__call__(messages, tools=tools, label=label, **kwargs)


def runtime_for(model):
    return VolumeReviewRuntime(
        call_model=model,
        parse_tool_arguments=lambda _name, raw: (json.loads(raw), ""),
        validate_speaker=valid_speaker,
        is_valid_speaker=lambda value: bool(valid_speaker(value)),
        same_speaker=lambda left, right: left == right,
        canonicalize_speaker=lambda value: (value, ""),
        is_generic_speaker=lambda value: value in {"少年", "店员", "路人"},
        retries=1,
        task_retries=1,
        task_retry_delay=0,
        sleep=lambda _seconds: None,
    )


def strong_verdict(speaker, *, basis="explicit_attribution", quote_type="direct_speech"):
    return {
        "id": "D0",
        "speaker": speaker,
        "quote_type": quote_type,
        "evidence_basis": basis,
        "confidence": "high",
        "citations": [1],
        "reason": "Local source evidence supports this assignment.",
    }


def mirrored_records(speakers_by_role):
    records = []
    for jury_role, _instruction in JURY_SPECS:
        values = speakers_by_role[jury_role]
        for (orientation, _reverse), speaker in zip(JURY_ORIENTATIONS, values):
            records.append(
                {
                    "jury_role": jury_role,
                    "orientation": orientation,
                    "verdict": strong_verdict(speaker),
                }
            )
    return records


class VolumeReviewStructureTests(unittest.TestCase):
    def test_json_extractor_does_not_promote_nested_citation_arrays(self):
        payload = {
            "assignments": [
                {
                    "id": "D0",
                    "speaker": "角色甲",
                    "citations": [1, 2],
                }
            ]
        }

        candidates = _structured_payloads_from_text(
            json.dumps(payload, ensure_ascii=False),
            "submit_volume_assignments",
        )

        self.assertEqual([payload], candidates)

    def test_review_units_overlap_context_but_not_focus(self):
        units = build_review_units(19, focus_size=8, context_radius=3)

        self.assertEqual((0, 8), (units[0]["focus_start"], units[0]["focus_end"]))
        self.assertEqual((8, 16), (units[1]["focus_start"], units[1]["focus_end"]))
        self.assertEqual(5, units[1]["context_start"])
        self.assertEqual(19, units[1]["context_end"])

    def test_scene_review_units_split_at_a_strong_narrative_gap(self):
        novel = ["角色甲说：「第一句。」"]
        novel.extend(["叙述。"] * 20)
        novel.append("角色乙说：「第二句。」")
        dialogues = [(1, "第一句。"), (22, "第二句。")]

        units = build_scene_review_units(
            novel,
            dialogues,
            focus_size=8,
            context_radius=1,
        )

        self.assertEqual(2, len(units))
        self.assertEqual((0, 1), (units[0]["focus_start"], units[0]["focus_end"]))
        self.assertEqual((1, 2), (units[1]["focus_start"], units[1]["focus_end"]))
        self.assertTrue(all(str(unit["unit_id"]).startswith("R2-S") for unit in units))

    def test_packet_marks_exact_focus_and_provisional_visibility(self):
        novel = ["角色甲说：「第一句。」角色乙回答：「第二句。」"]
        dialogues = [(1, "第一句。"), (1, "第二句。")]
        occurrences = align_quote_occurrences(novel, dialogues)
        unit = build_review_units(2, focus_size=1, context_radius=1)[1]

        blind, focus_ids, _ = format_review_packet(
            novel, occurrences, ["角色甲", "角色乙"], unit, include_provisional=False
        )
        visible, _, _ = format_review_packet(
            novel, occurrences, ["角色甲", "角色乙"], unit, include_provisional=True
        )

        self.assertEqual({"D1"}, focus_ids)
        self.assertIn("D0 [CONTEXT]", blind)
        self.assertIn("D1 [FOCUS]", blind)
        self.assertNotIn("provisional=", blind)
        self.assertIn("provisional=角色乙", visible)

    def test_hesitation_before_other_self_introduction_keeps_prior_floor(self):
        novel = [
            "「先前的话。」",
            "角色甲笑着说道。",
            "「那么，多多指教……呃——」",
            "「我叫角色乙。」",
        ]
        dialogues = [(1, "先前的话。"), (3, "那么，多多指教……呃——"), (4, "我叫角色乙。")]
        occurrences = align_quote_occurrences(novel, dialogues)

        constraints = derive_deterministic_sequence_constraints(
            novel,
            occurrences,
            ["角色甲", "角色乙", "角色乙"],
            is_valid_speaker=lambda value: bool(value),
            same_speaker=lambda left, right: left == right,
        )

        self.assertEqual("角色甲", constraints["D1"]["speaker"])
        self.assertEqual(
            "hesitation-before-other-self-introduction",
            constraints["D1"]["kind"],
        )

    def test_departure_farewell_is_assigned_to_other_party(self):
        novel = [
            "「谢谢您的邀请。」",
            "「那么，我们先告辞了。」",
            "「祝您一路顺风。」",
            "角色甲起身离开大厅。",
            "「下一段对话。」",
        ]
        dialogues = [
            (1, "谢谢您的邀请。"),
            (2, "那么，我们先告辞了。"),
            (3, "祝您一路顺风。"),
            (5, "下一段对话。"),
        ]
        occurrences = align_quote_occurrences(novel, dialogues)

        constraints = derive_deterministic_sequence_constraints(
            novel,
            occurrences,
            ["角色乙", "角色甲", "角色甲", "角色丙"],
            is_valid_speaker=lambda value: bool(value),
            same_speaker=lambda left, right: left == right,
        )

        self.assertEqual("角色乙", constraints["D2"]["speaker"])
        self.assertEqual(
            "departure-and-farewell-reciprocity",
            constraints["D2"]["kind"],
        )


class VolumeReviewDecisionGateTests(unittest.TestCase):
    def make_reviewer(self, root):
        return VolumeSecondPassReviewer(
            runtime_for(FakeModel({"D0": "角色甲"})),
            novel_path=str(root / "novel.txt"),
            labeled_path=str(root / "labeled.txt"),
            vault_path=str(root / "evidence_vault.json"),
            review_log_path=str(root / "volume_review.jsonl"),
            temp_log_path=str(root / "volume_review.temp.jsonl"),
            state_path=str(root / "volume_review_state.json"),
            first_pass_path=str(root / "labeled.first_pass.txt"),
        )

    def test_mirrored_jury_rejects_order_unstable_support(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reviewer = self.make_reviewer(Path(temp_dir))
            speakers = {}
            for index, (jury_role, _instruction) in enumerate(JURY_SPECS):
                if index < 2:
                    speakers[jury_role] = ("新标签", "新标签")
                elif index == 2:
                    speakers[jury_role] = ("新标签", "旧标签")
                else:
                    speakers[jury_role] = ("旧标签", "旧标签")

            assessment = reviewer._assess_mirrored_jury(
                "旧标签",
                "新标签",
                mirrored_records(speakers),
            )

            self.assertFalse(assessment["passes_general_gate"])
            self.assertEqual(2, assessment["candidate_pair_support"])
            self.assertEqual(1, assessment["baseline_pair_support"])
            self.assertEqual(1, len(assessment["unstable_roles"]))

    def test_three_of_four_stable_mirrored_roles_can_pass(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reviewer = self.make_reviewer(Path(temp_dir))
            speakers = {
                jury_role: (
                    ("新标签", "新标签")
                    if index < 3 else ("旧标签", "旧标签")
                )
                for index, (jury_role, _instruction) in enumerate(JURY_SPECS)
            }

            assessment = reviewer._assess_mirrored_jury(
                "旧标签",
                "新标签",
                mirrored_records(speakers),
            )

            self.assertTrue(assessment["passes_general_gate"])
            self.assertEqual(3, assessment["candidate_pair_support"])

    def test_named_to_generic_requires_all_pairs_and_direct_binding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reviewer = self.make_reviewer(Path(temp_dir))
            all_candidate = {
                jury_role: ("少年", "少年")
                for jury_role, _instruction in JURY_SPECS
            }
            assessment = reviewer._assess_mirrored_jury(
                "角色甲",
                "少年",
                mirrored_records(all_candidate),
            )
            candidate_group = {
                "speaker": "少年",
                "evidence_families": ["chronology", "quote-scope"],
                "strong_reports": [
                    {"agent": "VolumeBlindForward", **strong_verdict("少年")},
                    {"agent": "VolumeQuoteScope", **strong_verdict("少年")},
                ],
            }

            blocked, reason = reviewer._transition_gate(
                baseline="角色甲",
                candidate="少年",
                candidate_group=candidate_group,
                jury_assessment=assessment,
                novel_lines=["门被推开了。"],
                target_line=1,
                canonical_note="",
            )
            wrong_quote_bound, wrong_quote_reason = reviewer._transition_gate(
                baseline="角色甲",
                candidate="少年",
                candidate_group=candidate_group,
                jury_assessment=assessment,
                novel_lines=["少年说道：「前一句。」角色甲说道：「目标句。」"],
                target_line=1,
                canonical_note="",
                occurrence={
                    "scope_before": "角色甲说道：",
                    "scope_after": "",
                },
            )
            allowed, allowed_reason = reviewer._transition_gate(
                baseline="角色甲",
                candidate="少年",
                candidate_group=candidate_group,
                jury_assessment=assessment,
                novel_lines=["少年说道：「目标句。」"],
                target_line=1,
                canonical_note="",
            )

            self.assertFalse(blocked)
            self.assertIn("direct", reason)
            self.assertFalse(wrong_quote_bound)
            self.assertIn("direct", wrong_quote_reason)
            self.assertTrue(allowed)
            self.assertIn("direct", allowed_reason)

    def test_nonperson_change_requires_both_boundary_specialists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reviewer = self.make_reviewer(Path(temp_dir))
            all_nonperson = {
                jury_role: ("非人物发声", "非人物发声")
                for jury_role, _instruction in JURY_SPECS
            }
            assessment = reviewer._assess_mirrored_jury(
                "角色甲",
                "非人物发声",
                mirrored_records(all_nonperson),
            )
            scope_report = {
                "agent": "VolumeQuoteScope",
                **strong_verdict(
                    "非人物发声",
                    basis="quote_type",
                    quote_type="thought_or_narration",
                ),
            }
            candidate_group = {
                "speaker": "非人物发声",
                "evidence_families": ["quote-scope", "voice-boundary"],
                "strong_reports": [scope_report],
            }

            blocked, _reason = reviewer._transition_gate(
                baseline="角色甲",
                candidate="非人物发声",
                candidate_group=candidate_group,
                jury_assessment=assessment,
                novel_lines=["「目标句。」"],
                target_line=1,
                canonical_note="",
            )
            candidate_group["strong_reports"].append(
                {
                    "agent": "VolumeVoiceBoundary",
                    **strong_verdict(
                        "非人物发声",
                        basis="quote_type",
                        quote_type="thought_or_narration",
                    ),
                }
            )
            allowed, _allowed_reason = reviewer._transition_gate(
                baseline="角色甲",
                candidate="非人物发声",
                candidate_group=candidate_group,
                jury_assessment=assessment,
                novel_lines=["「目标句。」"],
                target_line=1,
                canonical_note="",
            )

            self.assertFalse(blocked)
            self.assertTrue(allowed)


class VolumeReviewRunTests(unittest.TestCase):
    def make_reviewer(self, root, model, *, retries=1, progress_callback=None):
        runtime = runtime_for(model)
        runtime.retries = retries
        return VolumeSecondPassReviewer(
            runtime,
            novel_path=str(root / "novel.txt"),
            labeled_path=str(root / "labeled.txt"),
            vault_path=str(root / "evidence_vault.json"),
            review_log_path=str(root / "volume_review.jsonl"),
            temp_log_path=str(root / "volume_review.temp.jsonl"),
            state_path=str(root / "volume_review_state.json"),
            first_pass_path=str(root / "labeled.first_pass.txt"),
            focus_size=2,
            context_radius=1,
            progress_callback=progress_callback,
        )

    def test_full_review_is_atomic_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root.joinpath("novel.txt").write_text(
                "角色甲说：「第一句。」\n角色乙回答：「第二句。」\n",
                encoding="utf-8",
            )
            root.joinpath("labeled.txt").write_text("角色甲\n角色乙\n", encoding="utf-8")
            root.joinpath("evidence_vault.json").write_text(
                '{"characters": {}}', encoding="utf-8"
            )
            model = FakeModel({"D0": "角色甲", "D1": "角色乙"})
            reviewer = self.make_reviewer(root, model)
            dialogues = [(1, "第一句。"), (2, "第二句。")]

            result = reviewer.run(dialogues)
            calls_after_first = model.calls
            repeated = reviewer.run(dialogues)

            self.assertEqual("complete", result["status"])
            self.assertEqual("already-complete", repeated["status"])
            self.assertEqual(calls_after_first, model.calls)
            self.assertEqual(
                "角色甲\n角色乙\n",
                root.joinpath("labeled.first_pass.txt").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                "角色甲\n角色乙\n",
                root.joinpath("labeled.txt").read_text(encoding="utf-8"),
            )

    def test_review_reports_durable_phase_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root.joinpath("novel.txt").write_text(
                "角色甲说：「第一句。」\n", encoding="utf-8"
            )
            root.joinpath("labeled.txt").write_text("角色甲\n", encoding="utf-8")
            root.joinpath("evidence_vault.json").write_text(
                '{"characters": {}}', encoding="utf-8"
            )
            events = []
            reviewer = self.make_reviewer(
                root,
                FakeModel({"D0": "角色甲"}),
                progress_callback=events.append,
            )

            reviewer.run([(1, "第一句。")])

            scene_updates = [
                event for event in events
                if event.get("event") == "task_complete"
                and event.get("phase") == "scene-review"
            ]
            self.assertEqual(9, len(scene_updates))
            self.assertEqual((9, 9), (scene_updates[-1]["completed"], scene_updates[-1]["total"]))
            state = json.loads(root.joinpath("volume_review_state.json").read_text(encoding="utf-8"))
            self.assertEqual("complete", state["progress"]["phase"])

    def test_unanimous_review_and_jury_can_rewrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root.joinpath("novel.txt").write_text(
                "新标签说：「目标句。」\n", encoding="utf-8"
            )
            root.joinpath("labeled.txt").write_text("旧标签\n", encoding="utf-8")
            root.joinpath("evidence_vault.json").write_text(
                '{"characters": {}}', encoding="utf-8"
            )
            model = FakeModel({"D0": "新标签"})
            reviewer = self.make_reviewer(root, model)

            result = reviewer.run([(1, "目标句。")])

            self.assertEqual(1, result["summary"]["changed"])
            self.assertEqual("新标签\n", root.joinpath("labeled.txt").read_text(encoding="utf-8"))
            self.assertEqual("旧标签\n", root.joinpath("labeled.first_pass.txt").read_text(encoding="utf-8"))

    def test_interrupted_review_never_overwrites_first_pass_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root.joinpath("novel.txt").write_text(
                "角色甲说：「第一句。」\n角色乙回答：「第二句。」\n",
                encoding="utf-8",
            )
            original = "角色甲\n角色乙\n"
            root.joinpath("labeled.txt").write_text(original, encoding="utf-8")
            root.joinpath("evidence_vault.json").write_text(
                '{"characters": {}}', encoding="utf-8"
            )
            model = InterruptingModel({"D0": "角色甲", "D1": "角色乙"})
            reviewer = self.make_reviewer(root, model)

            with self.assertRaisesRegex(KeyboardInterrupt, "simulated user interruption"):
                reviewer.run([(1, "第一句。"), (2, "第二句。")])

            self.assertEqual(original, root.joinpath("labeled.txt").read_text(encoding="utf-8"))
            self.assertEqual(
                original,
                root.joinpath("labeled.first_pass.txt").read_text(encoding="utf-8"),
            )
            state = json.loads(root.joinpath("volume_review_state.json").read_text(encoding="utf-8"))
            self.assertFalse(state["completed"])

    def test_permanently_bad_reviewer_abstains_without_aborting_volume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root.joinpath("novel.txt").write_text(
                "角色甲说：「第一句。」\n", encoding="utf-8"
            )
            root.joinpath("labeled.txt").write_text("角色甲\n", encoding="utf-8")
            root.joinpath("evidence_vault.json").write_text(
                '{"characters": {}}', encoding="utf-8"
            )
            reviewer = self.make_reviewer(
                root,
                AlwaysFailModel({"D0": "角色甲"}),
            )

            result = reviewer.run([(1, "第一句。")])

            self.assertEqual("complete", result["status"])
            self.assertEqual(9, result["summary"]["unit_abstentions"])
            self.assertEqual(0, result["summary"]["changed"])
            self.assertEqual(
                "角色甲\n",
                root.joinpath("labeled.txt").read_text(encoding="utf-8"),
            )

    def test_split_batch_resume_reuses_completed_single_items(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, content in {
                "novel.txt": "角色甲说：「第一句。」\n角色乙说：「第二句。」\n",
                "labeled.txt": "角色甲\n角色乙\n",
                "evidence_vault.json": '{"characters": {}}',
            }.items():
                root.joinpath(name).write_text(content, encoding="utf-8")
            first_model = FailOnSecondRecoveryTargetModel(
                {"D0": "角色甲", "D1": "角色乙"}
            )
            first = self.make_reviewer(root, first_model)
            user_content = """[Exact quote ledger]
D0 [FOCUS] L1 quote#1: 「第一句。」
D1 [FOCUS] L2 quote#1: 「第二句。」

[Raw novel excerpt]
L1: 角色甲说：「第一句。」
L2: 角色乙说：「第二句。」
"""
            validator = lambda args: first._validate_assignments(
                args, {"D0", "D1"}, {1, 2}
            )

            with self.assertRaisesRegex(RuntimeError, "second-item interruption"):
                first._recover_assignments_individually(
                    agent="VolumeBlindForward",
                    system_prompt=first._review_system_prompt(
                        "VolumeBlindForward", "Reconstruct chronology."
                    ),
                    user_content=user_content,
                    validator=validator,
                    record_meta={"kind": "unit", "unit_id": "U-test"},
                    required_ids={"D0", "D1"},
                    allowed_lines={1, 2},
                    batch_error="simulated malformed batch",
                )

            second_model = FakeModel({"D0": "角色甲", "D1": "角色乙"})
            second = self.make_reviewer(root, second_model)
            recovered = second._recover_assignments_individually(
                agent="VolumeBlindForward",
                system_prompt=second._review_system_prompt(
                    "VolumeBlindForward", "Reconstruct chronology."
                ),
                user_content=user_content,
                validator=lambda args: second._validate_assignments(
                    args, {"D0", "D1"}, {1, 2}
                ),
                record_meta={"kind": "unit", "unit_id": "U-test"},
                required_ids={"D0", "D1"},
                allowed_lines={1, 2},
                batch_error="resume",
            )

            self.assertEqual(1, second_model.calls)
            self.assertEqual(
                ["D0", "D1"],
                [item["id"] for item in recovered["assignments"]],
            )

    def test_failed_batch_is_recovered_by_single_target_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root.joinpath("novel.txt").write_text(
                "角色甲说：「第一句。」\n", encoding="utf-8"
            )
            root.joinpath("labeled.txt").write_text("角色甲\n", encoding="utf-8")
            root.joinpath("evidence_vault.json").write_text(
                '{"characters": {}}', encoding="utf-8"
            )
            model = ProseThenToolModel({"D0": "角色甲"})
            reviewer = self.make_reviewer(root, model)

            result = reviewer.run([(1, "第一句。")])

            self.assertEqual("complete", result["status"])
            self.assertEqual(18, model.calls)
            rejected = [
                json.loads(line)
                for line in root.joinpath("volume_review.temp.jsonl").read_text(encoding="utf-8").splitlines()
                if '"review_call_rejected"' in line
            ]
            self.assertEqual(9, len(rejected))
            split_completions = [
                json.loads(line)
                for line in root.joinpath("volume_review.temp.jsonl").read_text(encoding="utf-8").splitlines()
                if '"review_batch_split_complete"' in line
            ]
            self.assertEqual(9, len(split_completions))

    def test_failed_single_verdict_is_recovered_with_compact_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root.joinpath("novel.txt").write_text(
                "角色甲说：「第一句。」\n", encoding="utf-8"
            )
            root.joinpath("labeled.txt").write_text("角色甲\n", encoding="utf-8")
            root.joinpath("evidence_vault.json").write_text(
                '{"characters": {}}', encoding="utf-8"
            )
            model = ProseVerdictThenCompactModel({"D0": "角色甲"})
            reviewer = self.make_reviewer(root, model)
            user_content = """[Exact quote ledger]
D0 [FOCUS] L1 quote#1 | provisional=角色甲: 「第一句。」
  before=角色甲说：
  after=(none)

[Raw novel excerpt L1-L1]
L1: 角色甲说：「第一句。」

[Single target]
D0; provisional=角色甲; supported alternative=角色乙
[Independent reviewer reports]
- reviewer: speaker=角色甲; citations=[1]
"""

            result = reviewer._call_structured(
                agent="CompactTestJury",
                system_prompt=reviewer._jury_system_prompt(
                    "CompactTestJury",
                    "Audit exact quote scope.",
                ),
                user_content=user_content,
                tool=VERDICT_TOOL,
                expected_name="submit_volume_verdict",
                validator=lambda args: reviewer._validate_verdict(args, "D0", {1}),
                record_meta={"kind": "jury", "dialogue_id": "D0"},
            )

            self.assertEqual("角色甲", result["speaker"])
            self.assertEqual(2, model.calls)
            events = root.joinpath("volume_review.temp.jsonl").read_text(encoding="utf-8")
            self.assertIn('"review_compact_verdict_complete"', events)

    def test_compact_prose_verdict_is_recovered_by_json_compiler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root.joinpath("novel.txt").write_text(
                "角色甲说：「第一句。」\n", encoding="utf-8"
            )
            root.joinpath("labeled.txt").write_text("角色甲\n", encoding="utf-8")
            root.joinpath("evidence_vault.json").write_text(
                '{"characters": {}}', encoding="utf-8"
            )
            model = ProseVerdictWithJsonCompilerModel({"D0": "角色甲"})
            reviewer = self.make_reviewer(root, model)
            user_content = """[Exact quote ledger]
D0 [FOCUS] L1 quote#1 | provisional=角色甲: 「第一句。」
  before=角色甲说：
  after=(none)

[Raw novel excerpt L1-L1]
L1: 角色甲说：「第一句。」

[Single target]
D0; provisional=角色甲; supported alternative=角色乙
[Independent reviewer reports]
- reviewer: speaker=角色甲; citations=[1]
"""

            result = reviewer._call_structured(
                agent="CompactCompilerTestJury",
                system_prompt=reviewer._jury_system_prompt(
                    "CompactCompilerTestJury",
                    "Audit exact quote scope.",
                ),
                user_content=user_content,
                tool=VERDICT_TOOL,
                expected_name="submit_volume_verdict",
                validator=lambda args: reviewer._validate_verdict(args, "D0", {1}),
                record_meta={"kind": "jury", "dialogue_id": "D0"},
            )

            self.assertEqual("角色甲", result["speaker"])
            self.assertEqual(3, model.calls)
            events = root.joinpath("volume_review.temp.jsonl").read_text(encoding="utf-8")
            self.assertIn('"review_compact_serializer_complete"', events)
            self.assertIn('"result_source": "compact-text-json-serializer"', events)


if __name__ == "__main__":
    unittest.main()
