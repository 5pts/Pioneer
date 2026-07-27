"""Unit tests for prompt serialization, label parsing, and two-stage routing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from run import (  # noqa: E402
    LEVEL1_LABELS,
    LEVEL2_LABELS,
    build_messages,
    parse_label,
    route_final,
)


class RunTests(unittest.TestCase):
    """Check the deterministic parts of the API runner."""
    def test_parse_json_labels(self) -> None:
        self.assertEqual(parse_label('{"label":"W"}', LEVEL1_LABELS), "W")
        self.assertEqual(parse_label('{"label":"C"}', LEVEL2_LABELS), "C")
        self.assertIsNone(parse_label('{"label":"B"}', LEVEL1_LABELS))

    def test_routing(self) -> None:
        self.assertEqual(route_final("W", "Y"), "Y")
        self.assertEqual(route_final("R", None), "R")
        self.assertEqual(route_final("N", None), "N")
        self.assertIsNone(route_final("W", None))

    def test_few_shot_serialization(self) -> None:
        messages = build_messages(
            "prompt",
            [{"text": "示例文本", "gold_label": "B"}],
            "测试文本",
        )
        self.assertEqual(messages[2]["content"], '{"label": "B"}')
        self.assertIn("测试文本", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()

