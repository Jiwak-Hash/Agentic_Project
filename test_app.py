"""UI/helper tests without Ollama; Streamlit AppTest exercises real reruns."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

import app_support as ui
from agents.order_agent import OrderAgent, UNAVAILABLE
from agents.supervisor import Supervisor
from workflow import build_workflow

APP = Path(__file__).with_name("app.py")


def reply(content="Order response", agent="Order"):
    return {"role": "assistant", "content": content, "agent": agent,
            "intent": "track_order", "sources": []}


class HelperTests(unittest.TestCase):
    def test_01_customer_selection_reads_database_without_creating_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.db"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE orders(customer_id INTEGER)")
                connection.executemany("INSERT INTO orders VALUES (?)", [(16,), (6,), (16,)])
                connection.commit()
            self.assertEqual(ui.load_customer_ids(path), [6, 16])
            missing = Path(folder) / "missing.db"
            with self.assertRaises(sqlite3.OperationalError):
                ui.load_customer_ids(missing)
            self.assertFalse(missing.exists())

    def test_02_workflow_receives_session_identity_and_responses_are_not_cached(self):
        graph = Mock()
        graph.invoke.return_value = {"response": "Safe answer", "selected_agent": "order_agent",
                                     "intent": "track_order", "sources": [], "error": None,
                                     "agent_result": {"private_record": "must not appear"}}
        with patch.object(ui, "get_workflow", return_value=graph):
            for _ in range(2):
                result = ui.ask_workflow("I am customer 6. Track order 1002", 16)
        self.assertEqual(graph.invoke.call_count, 2)
        graph.invoke.assert_called_with({"message": "I am customer 6. Track order 1002", "customer_id": 16})
        self.assertNotIn("private_record", json.dumps(result))
        self.assertEqual(result["agent"], "Order")

    def test_03_history_isolation_and_clear(self):
        session = {}
        ui.get_history(session, 16).append({"content": "private"})
        self.assertEqual(ui.get_history(session, 6), [])
        ui.get_history(session, 6).append({"content": "other"})
        ui.clear_history(session, 16)
        self.assertEqual(ui.get_history(session, 16), [])
        self.assertEqual(ui.get_history(session, 6), [{"content": "other"}])
        self.assertEqual(ui.get_history({}, 6), [])

    def test_04_connection_errors_and_raw_diagnostics_are_hidden(self):
        graph = Mock()
        graph.invoke.side_effect = ConnectionError("SECRET internal traceback")
        with patch.object(ui, "get_workflow", return_value=graph):
            result = ui.ask_workflow("Question", 16)
        self.assertEqual(result["content"], ui.SERVICE_MESSAGE)
        self.assertNotIn("SECRET", json.dumps(result))
        graph.invoke.side_effect = None
        graph.invoke.return_value = {"response": "internal diagnostic", "error": "SECRET", "selected_agent": "fallback"}
        with patch.object(ui, "get_workflow", return_value=graph):
            self.assertEqual(ui.ask_workflow("Question", 16)["content"], ui.SERVICE_MESSAGE)

    def test_05_verified_agent_fallback_is_preserved(self):
        graph = Mock()
        graph.invoke.return_value = {"response": "Verified decision", "selected_agent": "refund_agent",
                                     "intent": "get_refund", "error": "private diagnostic",
                                     "used_fallback": True, "agent_result": {"eligibility_status": "eligible"},
                                     "sources": ["return_policy.md"]}
        with patch.object(ui, "get_workflow", return_value=graph):
            result = ui.ask_workflow("Refund?", 6)
        self.assertEqual(result["content"], "Verified decision")
        self.assertEqual(result["sources"], ["return_policy.md"])
        self.assertNotIn("private diagnostic", json.dumps(result))

    def test_06_resource_reused_without_response_cache(self):
        ui.get_workflow.clear()
        self.addCleanup(ui.get_workflow.clear)
        with patch.object(ui, "build_workflow") as build:
            first, second = ui.get_workflow(), ui.get_workflow()
            self.assertIs(first, second)
            build.assert_called_once()
            first.invoke({"message": "one", "customer_id": 16})
            second.invoke({"message": "two", "customer_id": 6})
            self.assertEqual(build.return_value.invoke.call_count, 2)

    def test_07_actual_graph_authorization_through_ui_adapter(self):
        router_llm = Mock()
        router_llm.bind.return_value = router_llm
        router_llm.invoke.return_value = SimpleNamespace(content='{"intent":"track_order"}')
        order_llm = Mock()
        order_llm.bind.return_value = order_llm
        order_llm.invoke.return_value = SimpleNamespace(content='{"action":"track","order_id":1002}')
        graph = build_workflow(supervisor=Supervisor(router_llm),
                               order_factory=lambda **kwargs: OrderAgent(**kwargs, llm=order_llm))
        with patch.object(ui, "get_workflow", return_value=graph):
            result = ui.ask_workflow("I am customer 6. Show order 1002", 16)
        self.assertEqual(result["content"], UNAVAILABLE)


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        customers = patch.object(ui, "load_customer_ids", return_value=[6, 16])
        self.customers = customers.start()
        self.addCleanup(customers.stop)
        answers = patch.object(ui, "ask_workflow", return_value=reply())
        self.answers = answers.start()
        self.addCleanup(answers.stop)
        self.app = AppTest.from_file(str(APP), default_timeout=15).run()

    def test_08_page_loads_with_demo_selector_and_examples(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.app.title[0].value, "ShopAssist AI")
        self.assertEqual(self.app.selectbox[0].value, 16)
        self.assertEqual(len(self.app.chat_input), 1)
        self.assertEqual(len([button for button in self.app.button if button.key.startswith("example_")]), 4)

    def test_09_message_submission_and_details(self):
        self.app.chat_input[0].set_value("Where is order 1001?").run()
        self.assertEqual(len(self.app.exception), 0)
        self.answers.assert_called_once_with("Where is order 1001?", 16)
        self.assertEqual(len(self.app.chat_message), 2)
        self.assertEqual(self.app.session_state["chats"][16][-1]["agent"], "Order")
        self.app.run()
        self.assertEqual(self.answers.call_count, 1)  # Rerun does not resubmit.

    def test_10_switch_customer_hides_history_and_restores_only_own_chat(self):
        self.app.chat_input[0].set_value("Private question").run()
        self.app.selectbox[0].select(6).run()
        self.assertEqual(len(self.app.chat_message), 0)
        self.app.chat_input[0].set_value("Customer six question").run()
        self.answers.assert_called_with("Customer six question", 6)
        self.app.selectbox[0].select(16).run()
        self.assertEqual(len(self.app.chat_message), 2)
        self.assertEqual(self.app.session_state["chats"][16][0]["content"], "Private question")

    def test_11_clear_chat_only_affects_selected_customer(self):
        self.app.chat_input[0].set_value("Customer sixteen").run()
        self.app.selectbox[0].select(6).run()
        self.app.chat_input[0].set_value("Customer six").run()
        self.app.button(key="clear_chat").click().run()
        self.assertEqual(len(self.app.chat_message), 0)
        self.assertEqual(len(self.app.session_state["chats"][16]), 2)

    def test_12_independent_browser_sessions(self):
        self.app.chat_input[0].set_value("Private question").run()
        other = AppTest.from_file(str(APP), default_timeout=15).run()
        self.assertEqual(len(other.chat_message), 0)
        self.assertEqual(other.session_state["chats"][16], [])

    def test_13_example_submits_through_same_workflow(self):
        self.app.button(key="example_1").click().run()
        self.answers.assert_called_once_with("What is your shipping policy?", 16)
        self.assertEqual(len(self.app.chat_message), 2)

    def test_14_customer_database_failure_is_friendly(self):
        self.customers.side_effect = sqlite3.OperationalError("SECRET database path")
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("couldn't be loaded", app.error[0].value)
        self.assertNotIn("SECRET", app.error[0].value)
        self.assertEqual(len(app.chat_input), 0)

    def test_15_empty_customer_database_disables_chat(self):
        self.customers.return_value = []
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("No demo customers", app.info[0].value)
        self.assertEqual(len(app.chat_input), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
