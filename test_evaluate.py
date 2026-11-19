"""Evaluator correctness tests without inference or edits to project datasets."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd

from evaluate import Checkpoints, LABELS, sample_dataset, order_checks
from evaluation.cases import FAQ_CASES, END_CASES, REFUND_CASES
from evaluation.reporting import class_metrics, review_for, timings


class EvaluationTests(unittest.TestCase):
    def test_stratification_is_balanced_unique_and_reproducible(self):
        frame = pd.DataFrame([dict(instruction=f"Question {label} {i}", intent=label)
                              for label in LABELS for i in range(25)])
        first = sample_dataset(frame, 20, 42)
        self.assertEqual(first, sample_dataset(frame, 20, 42))
        self.assertNotEqual(first, sample_dataset(frame, 20, 43))
        self.assertEqual(len(first), 200)
        self.assertEqual(len({row["dataset_row"] for row in first}), 200)
        self.assertEqual(set(pd.DataFrame(first).expected_intent.value_counts()), {20})

    def test_failed_prediction_stays_in_metric_denominator(self):
        metrics, report = class_metrics(["a", "b"], ["a", "__error__"], ["a", "b"])
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(metrics["macro_f1"], .5)
        self.assertEqual(report[1]["recall"], 0)

    def test_checkpoint_resume_does_not_repeat_or_drop_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            case = dict(id="one", question="Question")
            call = Mock(side_effect=RuntimeError("offline"))
            initial = Checkpoints(Path(folder), False).run("intent", case, call)
            self.assertIn("offline", initial["error"])
            reused = Checkpoints(Path(folder), True).run("intent", case, call)
            self.assertEqual(initial, reused)
            call.assert_called_once()
            with self.assertRaises(ValueError):
                Checkpoints(Path(folder), True).run("intent", dict(id="one", question="Changed"), call)

    def test_stale_review_is_rejected(self):
        with self.assertRaises(ValueError):
            review_for({"faq/a": {"response_sha256": "bad"}}, "faq", "a", "Answer")

    def test_order_evaluation_detects_invented_date(self):
        case = dict(kind="unknown", expected_records=[])
        result = dict(orders=[], answer="Your order will arrive on 2030-01-01.", error=None)
        checks = order_checks(case, result)
        self.assertFalse(checks["success"])
        self.assertFalse(checks["no_fabricated_dates"])

    def test_case_sizes_and_distinct_refund_outcomes(self):
        self.assertEqual(len(FAQ_CASES), 24)
        self.assertEqual(len(END_CASES), 20)
        self.assertEqual({row[-1] for row in REFUND_CASES}, {"eligible", "ineligible", "manual_review", "unknown"})

    def test_latency_statistics(self):
        result = timings([1, 2, 3, 4, 5])
        self.assertEqual(result["mean_seconds"], 3)
        self.assertEqual(result["median_seconds"], 3)
        self.assertAlmostEqual(result["p95_seconds"], 4.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
