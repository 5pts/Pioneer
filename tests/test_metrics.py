from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from evaluate import classification_metrics  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_perfect(self) -> None:
        rows = [
            {"gold": "B", "prediction": "B"},
            {"gold": "R", "prediction": "R"},
        ]
        result = classification_metrics(
            rows,
            labels=("B", "R"),
            gold_key="gold",
            prediction_key="prediction",
        )
        self.assertEqual(result["overall_accuracy"], 1.0)
        self.assertEqual(result["label_standardized_accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)

    def test_out_of_scope_prediction_is_error(self) -> None:
        rows = [{"gold": "B", "prediction": "N"}]
        result = classification_metrics(
            rows,
            labels=("B", "R"),
            gold_key="gold",
            prediction_key="prediction",
        )
        self.assertEqual(result["overall_accuracy"], 0.0)
        self.assertEqual(result["n_out_of_scope_or_missing_predictions"], 1)

    def test_standardization_equalizes_label_support(self) -> None:
        rows = [
            *[{"gold": "B", "prediction": "B"} for _ in range(9)],
            {"gold": "R", "prediction": "B"},
        ]
        result = classification_metrics(
            rows,
            labels=("B", "R"),
            gold_key="gold",
            prediction_key="prediction",
        )
        self.assertEqual(result["overall_accuracy"], 0.9)
        self.assertEqual(result["label_standardized_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()

