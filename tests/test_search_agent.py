import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from search_agent import SearchAgent  # noqa: E402


class SearchAgentTests(unittest.TestCase):
    def test_identity_search_enforces_long_forward_window(self):
        calls = []
        deep_search_calls = []

        def call_model(_messages, *, tools, label):
            calls.append((tools, label))
            if len(calls) == 1:
                return (
                    "",
                    10,
                    20,
                    [
                        {
                            "id": "fixture-call",
                            "type": "function",
                            "function": {
                                "name": "deep_search_identity",
                                "arguments": {
                                    "temp_name": "旅人",
                                    "around_line": 100,
                                    "search_forward": 200,
                                    "search_backward": 20,
                                },
                            },
                        }
                    ],
                )
            return '{"found": false, "character": "", "evidence": []}', 11, 21, []

        def deep_search(temp_name, around_line, search_forward, search_backward):
            deep_search_calls.append(
                (temp_name, around_line, search_forward, search_backward)
            )
            return "no identity found"

        agent = SearchAgent(
            call_model,
            lambda _start, _count: "",
            deep_search,
            lambda _name, _count: "",
        )
        result, pec, ec, _ = agent.investigate(
            "旅人",
            100,
            max_tool_rounds=2,
            quiet=True,
        )

        self.assertFalse(result["found"])
        self.assertEqual((21, 41), (pec, ec))
        self.assertEqual([("旅人", 100, 1200, 200)], deep_search_calls)

    def test_max_rounds_forces_structured_identity_submission(self):
        calls = []

        def call_model(_messages, *, tools, label, **_kwargs):
            calls.append(label)
            if label == "SearchAgent-R1":
                return (
                    "",
                    10,
                    20,
                    [
                        {
                            "id": "read-reveal",
                            "type": "function",
                            "function": {
                                "name": "read_novel_lines",
                                "arguments": {"start": 500, "count": 10},
                            },
                        }
                    ],
                )
            self.assertEqual("SearchAgent-Final", label)
            self.assertEqual("submit_identity_result", tools[0]["function"]["name"])
            return (
                "",
                30,
                40,
                [
                    {
                        "id": "submit-identity",
                        "type": "function",
                        "function": {
                            "name": "submit_identity_result",
                            "arguments": {
                                "found": True,
                                "character": "角色乙",
                                "aliases": ["旅人"],
                                "introduction_line": 505,
                                "status": "verified",
                                "evidence": [
                                    {"text": "旅人报出了姓名。", "line": 505, "type": "narrative_bridge"}
                                ],
                            },
                        },
                    }
                ],
            )

        agent = SearchAgent(
            call_model,
            lambda _start, _count: "500: 旅人继续前行。\n505: 旅人报出了姓名。",
            lambda *_args: "",
            lambda *_args: "",
        )
        result, pec, ec, _ = agent.investigate(
            "旅人",
            100,
            max_tool_rounds=1,
            quiet=True,
        )

        self.assertEqual(["SearchAgent-R1", "SearchAgent-Final"], calls)
        self.assertTrue(result["found"])
        self.assertEqual("verified", result["status"])
        self.assertEqual("角色乙", result["character"])
        self.assertEqual((40, 60), (pec, ec))


if __name__ == "__main__":
    unittest.main()
