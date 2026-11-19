# DemoCart FAQ Agent

This fictional company FAQ uses local MiniLM embeddings, FAISS, and the local
Ollama model `qwen2.5-coder:latest`. It does not import or access the order database.

From the project directory:

```bash
python -m pip install -r requirements.txt
# Start Ollama if it is not running: ollama serve
python -m rag.retriever
python test_faq.py
python -m unittest test_database.py
```

The first retrieval run downloads the free embedding model into `data/model_cache/`.
After that, embeddings run locally; set `HF_HUB_OFFLINE=1` to avoid Hub requests.
The FAISS index and readable JSON metadata are saved in `data/faiss_index/`.
Unchanged policies reuse the index; changed policies trigger a rebuild on the next
retriever initialization. Short policies are stored whole. If a future policy is
too long for MiniLM, split it into shorter Markdown files as the error instructs.

```python
from agents.faq_agent import FAQAgent

agent = FAQAgent()  # Reuse this instance across questions.
result = agent.answer("How long does delivery take?")
print(result["answer"])
print(result["sources"])
```

Missing/empty policies and model loading errors produce clear setup exceptions;
the test script prints these and exits with failure. Ollama failures return a
service-unavailable answer plus an `error` field and never count as passing tests.
Blank questions raise `ValueError`. Sources list all retrieved documents (top 3),
which may include documents the answer did not use.

`test_faq.py` prints all six requested questions, sources, answers, and checks,
and saves them to `data/faq_evaluation.json`. Evaluation checks expected-source
presence in the top 3 and manually specified answer facts, including abstention
on international shipping, which the policies deliberately leave unspecified.
This tiny keyword-based evaluation is not a general accuracy benchmark and does
not prove the absence of hallucinations. The grounding instruction is a prompt
constraint; review answers before using this demo in a real service.

API references: [SentenceTransformer](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
and [ChatOllama](https://reference.langchain.com/python/langchain-ollama/langchain_ollama/chat_models/ChatOllama).
