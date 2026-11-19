# ShopAssist AI

A local Streamlit chat interface for the existing LangGraph support workflow.
It uses the existing FAQ, Order, Refund, and Supervisor implementations directly.

## Start

From the project directory:

```bash
cd /home/jiwak/Agentic_Project
python -m pip install -r requirements.txt
```

Start Ollama in another terminal if it is not already running:

```bash
ollama serve
```

Check that the existing local model is available:

```bash
ollama list
```

The expected model is `qwen2.5-coder:latest`. If it is missing, download it with
`ollama pull qwen2.5-coder:latest`. Then start the interface:

```bash
streamlit run app.py
```

Open **http://127.0.0.1:8501**. The project configuration binds to localhost and
disables Streamlit usage telemetry. Stop the server with Ctrl+C.
Automatic file watching is disabled to avoid scanning PyTorch/Transformers
internals; restart Streamlit after changing application code.

The first policy request may take longer while MiniLM and FAISS load. The existing
model/index cache is reused. If the model is already cached, optional offline
Hugging Face mode avoids Hub checks:

```bash
HF_HUB_OFFLINE=1 streamlit run app.py
```

## Demo

Select **Demo customer ID** in the sidebar. IDs are read from the existing SQLite
database in read-only mode; Customer 16 is the default when present. This is a
demo identity selector, not authentication. All records are fictional. A future
authenticated application must obtain identity from its login/session system.

Try these questions in the input box or click an example:

- Customer 16: `Where is order 1001?`
- Any customer: `What is your shipping policy?`
- Customer 6: `Can I get a refund for order 1002?`
- Any customer: `How can I contact customer support?`
- Any customer: `List my orders`

Order examples work only for the owning demo account. Text claiming another
customer identity does not change the selected identity sent to LangGraph.
Refund checks use the backend's current-date default; the app does not add a new
eligibility rule or date override.

Each response shows the selected agent. Expand **Response details** for intent
and retrieved policy filenames, when present. Internal errors, raw order records,
prompts, and other workflow state are not rendered. Public policy sources are
retrieved context, not independently verified citations.

**Clear this chat** clears only the selected demo customer's history. Switching
customers hides the previous customer's conversation and restores the selected
customer's own conversation. Histories are scoped to the current browser session,
not shared through the resource cache. Reloading/disconnecting can reset a session.

## Implementation

- `app.py`: Streamlit layout, chat widgets, sidebar, loading indicator, and display.
- `app_support.py`: read-only customer ID query, session-history helpers, cached
  graph resource, and safe mapping from workflow output to display fields.
- `.streamlit/config.toml`: local server settings and teal/slate theme.
- `test_app.py`: helper tests and Streamlit AppTest interaction tests.

`ask_workflow()` passes only `message` and the selected `customer_id` to
`build_workflow().invoke()`. The UI does not classify intent or run agent logic.
The compiled graph is cached with `st.cache_resource`; its FAQ resource initializes
once and survives reruns. A lock serializes calls to protect shared lazy model
initialization. Concurrent demo visitors may therefore wait for another request.
Order/Refund customer-scoped agents continue to be created by the existing graph.
Neither customer responses nor histories are cached globally.

Connection and setup errors show a friendly retry message rather than technical
diagnostics. A specialist's verified deterministic fallback answer is preserved
when only wording generation failed.

## Tests

```bash
python test_app.py
python -m unittest -v test_database.py
python test_faq.py
python test_order_agent.py
python test_refund_agent.py --live
python test_workflow.py --live
```

UI tests mock inference, exercise actual Streamlit reruns, and include a real
LangGraph/Order Agent authorization check with mocked extraction. Live backend
tests require Ollama. Browser QA uses real local Chrome separately; Playwright is
not an application dependency.

## Limits

The interface keeps visible chat history, but the existing workflow handles each
message independently. Include the order ID again in follow-up questions. The app
does not execute order changes, cancellations, payments, or real human escalation.
Refund eligibility remains conditional request eligibility. Existing routing and
FAQ grounding limitations still apply; unusual wording may be misclassified.
This is a local semester demo, not a production authentication or ticketing system.

Streamlit references: [chat elements](https://docs.streamlit.io/develop/api-reference/chat),
[resource caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource),
and [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
