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
    VERDICT_TOOL,
    VolumeReviewRuntime,
    VolumeSecondPassReviewer,
    _structured_payloads_from_text,
    build_review_units,
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


class FailingModel(FakeModel):
    def __init__(self, speaker_by_id, fail_after):
        super().__init__(speaker_by_id)
        self.fail_after = fail_after

    def __call__(self, messages, *, tools, label, **kwargs):
        if self.calls >= self.fail_after:
            self.calls += 1
            raise RuntimeError("simulated provider interruption")
        return super().__call__(messages, tools=tools, label=label, **kwargs)


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


def runtime_for(model):
    return VolumeReviewRuntime(
        call_model=model,
        parse_tool_arguments=lambda _name, raw: (json.loads(raw), ""),
        validate_speaker=valid_speaker,
        is_valid_speaker=lambda value: bool(valid_speaker(value)),
        same_speaker=lambda left, right: left == right,
        canonicalize_speaker=lambda value: (value, ""),
        retries=1,
    )


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


class VolumeReviewRunTests(unittest.TestCase):
    def make_reviewer(self, root, model, *, retries=1):
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
            model = FailingModel({"D0": "角色甲", "D1": "角色乙"}, fail_after=1)
            reviewer = self.make_reviewer(root, model)

            with self.assertRaisesRegex(RuntimeError, "simulated provider interruption"):
                reviewer.run([(1, "第一句。"), (2, "第二句。")])

            self.assertEqual(original, root.joinpath("labeled.txt").read_text(encoding="utf-8"))
            self.assertEqual(
                original,
                root.joinpath("labeled.first_pass.txt").read_text(encoding="utf-8"),
            )
            state = json.loads(root.joinpath("volume_review_state.json").read_text(encoding="utf-8"))
            self.assertFalse(state["completed"])

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
            self.assertEqual(14, model.calls)
            rejected = [
                json.loads(line)
                for line in root.joinpath("volume_review.temp.jsonl").read_text(encoding="utf-8").splitlines()
                if '"review_call_rejected"' in line
            ]
            self.assertEqual(7, len(rejected))
            split_completions = [
                json.loads(line)
                for line in root.joinpath("volume_review.temp.jsonl").read_text(encoding="utf-8").splitlines()
                if '"review_batch_split_complete"' in line
            ]
            self.assertEqual(7, len(split_completions))

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
