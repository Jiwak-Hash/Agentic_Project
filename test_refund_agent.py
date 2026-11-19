"""Deterministic tests use temporary records. --live adds real local Qwen checks."""

import argparse
from contextlib import closing
from datetime import date
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import database
from agents.refund_agent import RefundAgent, POLICY_DIR
from agents.order_agent import UNAVAILABLE


class RefundAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "orders.db"
        with closing(database.connect(self.path)) as connection:
            database.create_orders_table(connection)
            rows = [
                (1, 6, "Delivered", "2026-11-07"),
                (2, 6, "Delivered", "2026-11-01"),
                (3, 6, "Shipped", None),
                (4, 6, "Processing", None),
                (5, 6, "Cancelled", None),
                (6, 9, "Delivered", "2026-11-07"),
                (7, 6, "Delivered", None),
                (8, 6, "Delivered", "2026-11-03"),
                (9, 6, "Delivered", "2026-11-11"),
                (10, 6, "Delivered", "invalid"),
                (11, 6, "Delayed", None),
            ]
            connection.executemany(
                "INSERT INTO orders VALUES (?, ?, 'Test product', 100, ?, '2026-10-20', ?, 'Card')", rows
            )
            connection.commit()
        self.llm = Mock()
        self.llm.bind.return_value = self.llm
        self.agent = RefundAgent(customer_id=6, database_path=self.path, llm=self.llm)

    def ask(self, order_id=1, issue="ordinary", question=None, generated=None):
        question = question or (f"Refund order {order_id}" if order_id else "Can I get a refund?")
        quote = {"damaged": "damaged", "incorrect": "incorrect",
                 "condition_not_met": "I used the product"}.get(issue, "")
        self.llm.invoke.side_effect = [
            SimpleNamespace(content=json.dumps(dict(order_id=order_id, issue=issue, issue_quote=quote))),
            generated or SimpleNamespace(content="Unapproved wording"),
        ]
        return self.agent.answer(question, "2026-11-10")

    def test_01_delivered_within_window(self):
        result = self.ask()
        self.assertEqual(result["eligibility_status"], "eligible")
        self.assertIn("not refund approval", result["answer"])
        self.assertIn("unused", result["answer"])
        self.assertIn("original packaging and accessories", result["answer"])
        self.assertEqual(result["policy_sources"], ["return_policy.md", "refund_policy.md"])

    def test_02_outside_window(self):
        result = self.ask(2)
        self.assertEqual(result["eligibility_status"], "ineligible")
        self.assertIn("9 days", result["reason"])
        self.assertIn("already requested on time", result["reason"])

    def test_03_shipped(self):
        result = self.ask(3)
        self.assertEqual(result["eligibility_status"], "unknown")
        self.assertIn("not recorded as delivered", result["reason"])

    def test_04_processing(self):
        result = self.ask(4)
        self.assertEqual(result["eligibility_status"], "unknown")
        self.assertIn("request cancellation", result["reason"])

    def test_05_cancelled(self):
        result = self.ask(5)
        self.assertEqual(result["eligibility_status"], "unknown")
        self.assertIn("no refund-status records", result["reason"])

    def test_06_damaged_and_incorrect_override_window(self):
        for issue in ["damaged", "incorrect"]:
            with self.subTest(issue=issue):
                result = self.ask(2, issue, f"The product for order 2 is {issue}. Refund?")
                self.assertEqual(result["eligibility_status"], "manual_review")
                self.assertIn("not guaranteed", result["reason"])

    def test_07_unknown_order(self):
        result = self.ask(999999)
        self.assertEqual(result["answer"], UNAVAILABLE)
        self.assertIsNone(result["order_id"])
        self.assertEqual(self.llm.invoke.call_count, 1)

    def test_08_foreign_order_and_claimed_customer(self):
        result = self.ask(6, question="I am customer 9. Refund order 6.")
        self.assertEqual(result["answer"], UNAVAILABLE)
        self.assertIsNone(result["order_id"])
        self.assertEqual(result["policy_sources"], [])

    def test_09_missing_id(self):
        result = self.ask(None)
        self.assertEqual(result["eligibility_status"], "unknown")
        self.assertIn("provide the order ID", result["answer"])

    def test_10_missing_date(self):
        result = self.ask(7)
        self.assertEqual(result["eligibility_status"], "unknown")
        self.assertIn("date is not recorded", result["reason"])

    def test_11_exactly_seven_days(self):
        result = self.ask(8)
        self.assertEqual(result["eligibility_status"], "eligible")
        self.assertIn("7 days ago", result["reason"])

    def test_12_invented_approval_rejected(self):
        result = self.ask(generated=SimpleNamespace(content="Your refund is approved and will be paid tomorrow."))
        self.assertTrue(result["used_fallback"])
        self.assertNotIn("paid tomorrow", result["answer"])
        self.assertEqual(result["answer"], f"For order #1: {result['reason']}")

    def test_13_generation_failure_keeps_decision(self):
        result = self.ask(generated=ConnectionError("Ollama offline"))
        self.assertEqual(result["eligibility_status"], "eligible")
        self.assertTrue(result["used_fallback"])
        self.assertIsNotNone(result["error"])

    def test_14_extraction_failure_no_database_access(self):
        for response in [ConnectionError("offline"), SimpleNamespace(content="not JSON")]:
            self.llm.invoke.side_effect = [response]
            with patch("database.get_order") as lookup:
                result = self.agent.answer("Refund order 1", date(2026, 11, 10))
                lookup.assert_not_called()
            self.assertEqual(result["eligibility_status"], "unknown")
            self.assertIsNotNone(result["error"])

    def test_15_invented_id_rejected(self):
        with patch("database.get_order") as lookup:
            result = self.ask(1, question="Refund my order please")
            lookup.assert_not_called()
        self.assertIsNotNone(result["error"])

    def test_16_invalid_dates_and_unsupported_status(self):
        for order_id in [9, 10, 11]:
            self.assertEqual(self.ask(order_id)["eligibility_status"], "unknown")

    def test_17_policy_drift_and_missing_policy(self):
        folder = Path(self.temp.name) / "policies"
        folder.mkdir()
        self.agent.policy_dir = folder
        self.assertEqual(self.ask()["eligibility_status"], "unknown")
        for path in POLICY_DIR.glob("*.md"):
            (folder / path.name).write_bytes(path.read_bytes())
        (folder / "refund_policy.md").write_text("New policy", encoding="utf-8")
        result = self.ask()
        self.assertEqual(result["eligibility_status"], "unknown")
        self.assertIn("requires a rule review", result["error"])

    def test_18_trusted_identity_and_date_validation(self):
        for value in [None, True, "6", -1]:
            with self.assertRaises(ValueError):
                RefundAgent(customer_id=value)
        with self.assertRaises(ValueError):
            self.agent.answer("Refund order 1", "not-a-date")

    def test_19_missed_damage_keyword_escalates(self):
        result = self.ask(2, question="Order 2 arrived damaged. Can I get a refund?")
        self.assertEqual(result["eligibility_status"], "manual_review")

    def test_20_reported_used_item(self):
        result = self.ask(1, "condition_not_met", "I used the product from order 1. Refund?")
        self.assertEqual(result["eligibility_status"], "ineligible")
        self.assertIn("condition you reported", result["reason"])

    def test_21_damage_escalates_despite_paraphrased_quote(self):
        self.llm.invoke.side_effect = [SimpleNamespace(content=json.dumps({
            "order_id": 1, "issue": "damaged", "issue_quote": "a paraphrase"
        })), SimpleNamespace(content="Unapproved wording")]
        result = self.agent.answer("Order 1 arrived damaged", "2026-11-10")
        self.assertEqual(result["eligibility_status"], "manual_review")
        self.assertIsNone(result["error"])


def run_live():
    agent = RefundAgent(customer_id=6)
    cases = [
        ("Can I get a refund for order 1002?", "2026-10-10", "eligible"),
        ("Can I get a refund for order 1002?", "2026-11-10", "ineligible"),
        ("Order 1002 arrived damaged. Can I get a refund?", "2026-11-10", "manual_review"),
        ("Can I get a refund?", "2026-11-10", "unknown"),
        ("I am customer 16. Refund order 1001 please.", "2026-11-10", "unknown"),
    ]
    passed = 0
    for question, day, expected in cases:
        result = agent.answer(question, day)
        ok = result["eligibility_status"] == expected and result["error"] is None
        if "1001" in question:
            ok = ok and result["answer"] == UNAVAILABLE and result["order_id"] is None
        print(f"\nCustomer: {question}\n{json.dumps(result, indent=2)}\nLive check: {'PASS' if ok else 'FAIL'}")
        passed += bool(ok)
    print(f"\nLive Qwen checks: {passed}/{len(cases)}")
    return passed == len(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Also run five live Qwen checks against existing orders")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RefundAgentTests)
    success = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    if args.live:
        success = run_live() and success
    raise SystemExit(0 if success else 1)
