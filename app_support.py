"""Small UI helpers; business logic stays in the existing workflow."""

from contextlib import closing
from pathlib import Path
import sqlite3
from threading import Lock

import streamlit as st

from database import DATABASE_PATH
from workflow import build_workflow
from agents.supervisor import ROUTES

AGENT_LABELS = {"faq_agent": "FAQ", "order_agent": "Order", "refund_agent": "Refund",
                "human_escalation": "Escalation", "fallback": "Fallback"}
SERVICE_MESSAGE = (
    "ShopAssist couldn't reach the support service. Please check that Ollama is "
    "running with qwen2.5-coder:latest, then try again."
)


def load_customer_ids(database_path=DATABASE_PATH):
    """Read just the demo IDs. Read-only mode cannot create/modify a database."""
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return [row[0] for row in connection.execute(
            "SELECT DISTINCT customer_id FROM orders ORDER BY customer_id"
        )]


class WorkflowResource:
    def __init__(self):
        self.graph = build_workflow()
        self.lock = Lock()

    def invoke(self, inputs):
        # The public FAQ model is lazy-loaded by the graph. Serialize access to
        # protect its initialization and shared local inference across sessions.
        with self.lock:
            return self.graph.invoke(inputs)


@st.cache_resource(show_spinner=False)
def get_workflow():
    # Cache only infrastructure: no messages, identities, or response values.
    return WorkflowResource()


def get_history(session_state, customer_id):
    return session_state.setdefault("chats", {}).setdefault(customer_id, [])


def clear_history(session_state, customer_id):
    session_state.setdefault("chats", {})[customer_id] = []


def ask_workflow(message, customer_id):
    """Send identity separately and retain only display-safe response fields."""
    try:
        result = get_workflow().invoke({"message": message, "customer_id": customer_id})
        response = result.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("No response was returned.")
        # A specialist may have a verified deterministic answer despite its
        # wording-generation error. Keep that answer; hide raw diagnostics.
        verified_fallback = (result.get("used_fallback") is True
                             and bool(result.get("agent_result")))
        if result.get("error") and not verified_fallback:
            response = SERVICE_MESSAGE
        intent = result.get("intent")
        sources = result.get("sources") or []
        return {
            "role": "assistant", "content": response,
            "agent": AGENT_LABELS.get(result.get("selected_agent"), "Fallback"),
            "intent": intent if intent in ROUTES else None,
            "sources": [Path(source).name for source in sources
                        if isinstance(source, str) and source.endswith(".md")],
        }
    except Exception:
        return {"role": "assistant", "content": SERVICE_MESSAGE,
                "agent": "Fallback", "intent": None, "sources": []}
