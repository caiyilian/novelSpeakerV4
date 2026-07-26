import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scene_sequence import (  # noqa: E402
    assignment_for_dialogue,
    build_scene_packet,
    build_scene_segments,
    find_scene_for_dialogue,
    format_scene_packet,
    normalize_scene_assignments,
)


class SceneSegmentationTests(unittest.TestCase):
    def test_every_target_in_a_segment_resolves_to_the_same_scene(self):
        lines = [f"narrative {index}\n" for index in range(1, 41)]
        dialogues = [(3, "a"), (5, "b"), (7, "c"), (9, "d")]
        segments = build_scene_segments(lines, dialogues, max_turns=10)

        self.assertEqual(1, len(segments))
        scene_ids = {
            find_scene_for_dialogue(segments, index)["scene_id"]
            for index in range(len(dialogues))
        }
        self.assertEqual({segments[0]["scene_id"]}, scene_ids)

    def test_heading_creates_a_strong_scene_boundary(self):
        lines = ["叙事\n" for _ in range(30)]
        lines[9] = "第二章\n"
        dialogues = [(5, "a"), (8, "b"), (12, "c"), (14, "d")]
        segments = build_scene_segments(lines, dialogues, max_turns=10)

        self.assertEqual([(0, 2), (2, 4)], [
            (segment["start_index"], segment["end_index"])
            for segment in segments
        ])

    def test_oversized_scene_is_split_at_a_stable_boundary(self):
        lines = [f"叙事 {index}\n" for index in range(1, 80)]
        dialogues = [(index * 2 + 1, f"d{index}") for index in range(10)]
        segments = build_scene_segments(lines, dialogues, max_turns=4, hard_gap_lines=20)

        self.assertTrue(all(segment["turn_count"] <= 4 for segment in segments))
        covered = [
            index
            for segment in segments
            for index in range(segment["start_index"], segment["end_index"])
        ]
        self.assertEqual(list(range(len(dialogues))), covered)


class ScenePacketTests(unittest.TestCase):
    def setUp(self):
        self.lines = [f"原文 {index}\n" for index in range(1, 31)]
        self.dialogues = [(5, "第一句"), (7, "第二句"), (7, "同一行第三句")]
        self.segment = build_scene_segments(self.lines, self.dialogues, max_turns=10)[0]
        self.packet = build_scene_packet(self.lines, self.dialogues, self.segment)

    @staticmethod
    def _speaker(value):
        return value if value and " " not in value else None

    def test_packet_preserves_same_line_occurrence_and_target(self):
        rendered = format_scene_packet(
            self.packet,
            target_dialogue_index=2,
            max_raw_chars=4000,
        )

        self.assertIn("D2 L7 occurrence#2 TARGET", rendered)
        self.assertIn("同一行第三句", rendered)
        self.assertIn(">> L7:", rendered)

    def test_complete_scene_map_is_normalized(self):
        raw = [
            {
                "turn_id": turn["id"],
                "line": turn["line"],
                "speaker": f"人物{index}",
                "quote_type": "direct_speech",
                "evidence_basis": "speaker_action",
                "confidence": "medium",
                "citations": [turn["line"]],
                "reason": "local evidence",
            }
            for index, turn in enumerate(self.packet["turns"], 1)
        ]

        normalized = normalize_scene_assignments(raw, self.packet, self._speaker)

        self.assertTrue(normalized["valid"])
        self.assertEqual("人物3", assignment_for_dialogue(normalized, self.packet, 2)["speaker"])

    def test_map_rejects_missing_turn_and_nonlocal_citation(self):
        first = self.packet["turns"][0]
        raw = [{
            "turn_id": first["id"],
            "line": first["line"],
            "speaker": "人物一",
            "citations": [9999],
        }]

        normalized = normalize_scene_assignments(raw, self.packet, self._speaker)

        self.assertFalse(normalized["valid"])
        self.assertIn("D1", normalized["missing"])
        self.assertTrue(any("scene-local citation" in error for error in normalized["errors"]))


if __name__ == "__main__":
    unittest.main()
