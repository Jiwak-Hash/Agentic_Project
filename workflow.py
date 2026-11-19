"""One supervisor, one destination, then END. Identity comes from the caller."""

from datetime import date
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.supervisor import ROUTES, Supervisor


class WorkflowInput(TypedDict):
    message: str
    customer_id: int  # Trusted session ID; never parsed from message text.
    evaluation_date: NotRequired[str | date | None]


class WorkflowState(WorkflowInput, total=False):
    intent: str
    selected_agent: str
    response: str
    sources: list[str]
    agent_result: dict
    error: str | None
    used_fallback: bool


CHANGE_LIMITATION = "This demo cannot change orders, addresses, or quantities. Please contact customer support to request a change. No order has been changed."
CANCEL_LIMITATION = "This demo cannot cancel orders. Please contact customer support to request cancellation and verify eligibility. No order has been cancelled."
REFUND_TRACKING_LIMITATION = "Refund tracking is unavailable because this demo has no refund-status or payment records. Please contact customer support to verify your refund status."
ESCALATION_RESPONSE = "This is a simulated human escalation. Please contact customer support directly. No support ticket has been created and no human agent has been contacted."
FALLBACK_RESPONSE = "I can help with order tracking, delivery and return policies, or refund eligibility. Please ask one supported question at a time."


def _agent_update(result):
    """Forward the specialist's validated answer without another LLM rewrite."""
    return {"response": result["answer"], "agent_result": result,
            "sources": result.get("sources", result.get("policy_sources", [])),
            "error": result.get("error"),
            "used_fallback": result.get("used_fallback", False)}


def _service_failure(error):
    return {"response": "The selected support service is temporarily unavailable. Please try again later.",
            "error": str(error), "used_fallback": True, "sources": [], "agent_result": {}}


def build_workflow(*, supervisor=None, faq_factory=None, order_factory=None,
                   refund_factory=None):
    """Factories allow isolated tests; only the selected specialist is created.

    Reuse the compiled graph across requests. No customer-specific agent instance
    or conversation state is cached across requests.
    """
    router = supervisor if supervisor is not None else Supervisor()
    faq_instance = None  # FAQ is public policy only; reuse its embedding model.

    def supervisor_node(state):
        message, customer_id = state.get("message"), state.get("customer_id")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Please provide a non-empty message.")
        if type(customer_id) is not int or not 0 < customer_id < 2**63:
            raise ValueError("A valid customer ID from the trusted session is required.")
        value = state.get("evaluation_date")
        if value is not None:
            if isinstance(value, str):
                date.fromisoformat(value)
            elif type(value) is not date:
                raise ValueError("evaluation_date must be a date or YYYY-MM-DD string.")
        result = router.classify(message)
        intent = result["intent"]
        # The model never writes the state, destination, customer ID, or message.
        return {"intent": intent, "selected_agent": ROUTES[intent],
                "error": result.get("error"), "response": "", "sources": [],
                "agent_result": {}, "used_fallback": False}

    def faq_node(state):
        nonlocal faq_instance
        try:
            if faq_instance is None:
                if faq_factory is None:
                    from agents.faq_agent import FAQAgent
                    faq_instance = FAQAgent()
                else:
                    faq_instance = faq_factory()
            return _agent_update(faq_instance.answer(state["message"]))
        except Exception as error:
            return _service_failure(error)

    def order_node(state):
        # These operations are intentionally not delegated as tracking requests.
        if state["intent"] == "change_order":
            return {"response": CHANGE_LIMITATION}
        if state["intent"] == "cancel_order":
            return {"response": CANCEL_LIMITATION}
        try:
            if order_factory is None:
                from agents.order_agent import OrderAgent
                agent = OrderAgent(customer_id=state["customer_id"])
            else:
                agent = order_factory(customer_id=state["customer_id"])
            return _agent_update(agent.answer(state["message"]))
        except Exception as error:
            return _service_failure(error)

    def refund_node(state):
        if state["intent"] == "track_refund":
            # Eligibility is not refund status. No record lookup is needed to
            # explain a capability limit, and no order information is revealed.
            return {"response": REFUND_TRACKING_LIMITATION}
        try:
            if refund_factory is None:
                from agents.refund_agent import RefundAgent
                agent = RefundAgent(customer_id=state["customer_id"])
            else:
                agent = refund_factory(customer_id=state["customer_id"])
            return _agent_update(agent.answer(
                state["message"], evaluation_date=state.get("evaluation_date")
            ))
        except Exception as error:
            return _service_failure(error)

    def human_node(state):
        return {"response": ESCALATION_RESPONSE}

    def fallback_node(state):
        response = ("I couldn't route your request right now. Please try again."
                    if state.get("error") else FALLBACK_RESPONSE)
        return {"response": response, "used_fallback": True}

    graph = StateGraph(WorkflowState, input_schema=WorkflowInput)
    graph.add_node("supervisor", supervisor_node)
    destinations = {"faq_agent": faq_node, "order_agent": order_node,
                    "refund_agent": refund_node, "human_escalation": human_node,
                    "fallback": fallback_node}
    for name, node in destinations.items():
        graph.add_node(name, node)
        graph.add_edge(name, END)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda state: state["selected_agent"],
                                {name: name for name in destinations})
    return graph.compile()
