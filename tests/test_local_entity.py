import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_entity import build_local_entity_consensus  # noqa: E402


def graph_record(agent, entities, assignments):
    return {
        "agent": agent,
        "graph": {"entities": entities, "assignments": assignments},
    }


def entity(entity_id, canonical, mentions):
    return {
        "entity_id": entity_id,
        "canonical_label": canonical,
        "mentions": mentions,
        "citations": [1],
    }


def assignment(dialogue_id, entity_id, speaker):
    return {"id": dialogue_id, "entity_id": entity_id, "speaker": speaker}


class LocalEntityConsensusTests(unittest.TestCase):
    def build(self, records):
        return build_local_entity_consensus(
            records,
            same_speaker=lambda left, right: left == right,
            is_generic_speaker=lambda value: value in {"店员", "年轻店员", "少年"},
            canonicalize_speaker=lambda value: (value, ""),
        )

    def test_two_of_three_graphs_merge_source_aliases(self):
        records = [
            graph_record(
                "forward",
                [entity("E1", "年轻店员", ["年轻店员", "店员"])],
                [assignment("D0", "E1", "年轻店员")],
            ),
            graph_record(
                "reverse",
                [entity("P1", "店员", ["店员", "年轻店员"])],
                [assignment("D0", "P1", "店员")],
            ),
            graph_record(
                "scope",
                [
                    entity("X1", "年轻店员", ["年轻店员"]),
                    entity("X2", "店员", ["店员"]),
                ],
                [assignment("D0", "X1", "年轻店员")],
            ),
        ]

        consensus = self.build(records)

        self.assertTrue(consensus.same_entity("年轻店员", "店员"))
        self.assertEqual(3, consensus.assignment_support("D0", "店员"))
        self.assertEqual(3, consensus.presence_support("年轻店员"))

    def test_rendering_preserves_baseline_for_same_local_entity(self):
        records = [
            graph_record(
                agent,
                [entity("E1", "年轻店员", ["年轻店员", "店员"])],
                [assignment("D0", "E1", "店员")],
            )
            for agent in ("forward", "reverse", "scope")
        ]
        consensus = self.build(records)

        speaker, note = consensus.canonical_for("年轻店员", baseline="店员")

        self.assertEqual("店员", speaker)
        self.assertEqual("local-entity-preserved-baseline", note)

    def test_rendering_upgrades_role_to_source_grounded_name(self):
        records = [
            graph_record(
                agent,
                [entity("E1", "角色甲", ["角色甲", "店员"])],
                [assignment("D0", "E1", "店员")],
            )
            for agent in ("forward", "reverse", "scope")
        ]
        consensus = self.build(records)

        speaker, note = consensus.canonical_for("店员")

        self.assertEqual("角色甲", speaker)
        self.assertEqual("local-entity-canonical-label", note)

    def test_competing_named_assignment_is_counted_separately(self):
        records = [
            graph_record(
                agent,
                [
                    entity("E1", "角色甲", ["角色甲"]),
                    entity("E2", "角色乙", ["角色乙"]),
                ],
                [assignment("D0", "E2", "角色乙")],
            )
            for agent in ("forward", "reverse", "scope")
        ]
        consensus = self.build(records)

        self.assertEqual(0, consensus.assignment_support("D0", "角色甲"))
        self.assertEqual(3, consensus.competing_named_support("D0", "角色甲"))


if __name__ == "__main__":
    unittest.main()
