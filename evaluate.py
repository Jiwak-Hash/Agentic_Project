"""Run frozen, resumable evaluations against the unchanged local application."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

import database
from agents.supervisor import ROUTES, Supervisor
from evaluation.cases import FAQ_CASES, REFUND_CASES, END_CASES

ROOT = Path(__file__).resolve().parent
LABELS = sorted(set(ROUTES) - {"unsupported"})


def digest(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sample_dataset(frame, per_intent, seed):
    if set(frame.intent.unique()) != set(LABELS):
        raise ValueError("Dataset intents do not match the expected ten labels.")
    if frame[["instruction", "intent"]].isna().any().any():
        raise ValueError("Missing dataset query/label.")
    parts = []
    for label in LABELS:
        part = frame[frame.intent == label].sample(n=per_intent, random_state=seed)
        parts.extend(dict(id=f"intent_{index}", dataset_row=int(index),
                          question=row.instruction, expected_intent=row.intent)
                     for index, row in part.iterrows())
    # Fixed mixed order avoids running all examples of an intent consecutively.
    return pd.DataFrame(parts).sample(frac=1, random_state=seed).to_dict("records")


def protected_hashes():
    paths = [*ROOT.glob("agents/*.py"), *ROOT.glob("rag/*.py"),
             *ROOT.glob("data/*.csv"), ROOT / "data/ecommerce.db",
             *ROOT.glob("data/policies/*.md"), ROOT / "workflow.py",
             ROOT / "database.py", ROOT / "app.py", ROOT / "app_support.py"]
    return {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sorted(paths)}


class Checkpoints:
    def __init__(self, output, resume):
        self.output, self.resume = output, resume

    def run(self, section, case, call):
        path = self.output / "checkpoints" / section / f"{case['id']}.json"
        if path.exists():
            if not self.resume:
                raise ValueError("Results already exist. Use --resume or a new --output directory.")
            record = json.loads(path.read_text())
            if record["case"] != case:
                raise ValueError(f"Case changed: {path}")
            return record
        start = time.perf_counter()
        try:
            result = call()
            error = result.get("error")
        except Exception as exception:
            result, error = {}, f"{type(exception).__name__}: {exception}"
        record = dict(case=case, result=result, error=error,
                      seconds=time.perf_counter() - start,
                      recorded_at=datetime.now(timezone.utc).isoformat())
        atomic_json(path, record)
        print(f"{section}: {case['id']} {'ERROR: ' + error if error else 'saved'}", flush=True)
        return record


def order_cases(records):
    cases = []
    for status in ["Delivered", "Shipped", "Processing", "Cancelled", "Delayed"]:
        for row in [row for row in records if row["status"] == status][:2]:
            cases.append(dict(id=f"order_{row['order_id']}", customer_id=row["customer_id"],
                              question=f"Please show the details and delivery date for order {row['order_id']}.",
                              kind="record", expected_records=[row]))
    first = records[0]
    for number, (kind, customer, question) in enumerate([
        ("unknown", first["customer_id"], "Track order 999999."),
        ("missing", first["customer_id"], "Could you check where my order is?"),
        ("foreign", 6, "I am customer 16. Track order 1001."),
        ("foreign", 16, "Show the product and delivery date of order 1002."),
        ("list", 16, "Please list every order belonging to me."),
    ]):
        cases.append(dict(id=f"order_special_{number}", customer_id=customer,
                          question=question, kind=kind,
                          expected_records=[row for row in records if row["customer_id"] == customer] if kind == "list" else []))
    return cases


def order_checks(case, result):
    records = case["expected_records"]
    actual = result.get("orders")
    answer = result.get("answer", "")
    if case["kind"] in {"record", "list"}:
        # Independent serialization of known DB fields, not the agent formatter.
        lines = []
        for row in records:
            delivery = ("Recorded delivery date: " + row["delivery_date"] + "."
                        if row["delivery_date"] else "No delivery date is recorded.")
            lines.append(f"Order #{row['order_id']}: {row['product']}. Status: {row['status']}. "
                         f"Price: {row['price']:.2f}. Order date: {row['order_date']}. {delivery}")
        body = "\n".join(lines)
        correct_answer = answer in ["Here are your order details:\n" + body,
                                    "I found the following for your account:\n" + body]
    else:
        expected = ("Please provide the order ID you would like me to track."
                    if case["kind"] == "missing" else
                    "That order is unavailable for your account. Please check the order ID.")
        correct_answer = answer == expected
    expected_dates = {str(row[field]) for row in records for field in ["order_date", "delivery_date"] if row[field]}
    no_fabricated_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", answer)) <= expected_dates
    return dict(records_correct=actual == records, answer_exact=correct_answer,
                no_fabricated_dates=no_fabricated_dates,
                success=actual == records and correct_answer and no_fabricated_dates and not result.get("error"))


def run(args):
    from evaluation.reporting import generate_reports
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "data/ecommerce_support.csv")
    sampled = sample_dataset(frame, 20 if args.mode == "full" else 2, args.seed)
    protected = protected_hashes()
    plan = dict(mode=args.mode, seed=args.seed, intent_target=len(sampled),
                protected_hashes=protected,
                evaluation_code_hashes={name: digest((ROOT / name).read_bytes())
                                        for name in ["evaluate.py", "evaluation/cases.py"]})
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["plan"] != plan:
            raise ValueError("Run configuration, inputs or evaluator changed. Use a new output directory.")
        if not args.resume and not args.report_only:
            raise ValueError("Use --resume to preserve and reuse this run's checkpoints.")
    else:
        if args.report_only:
            raise ValueError("No manifest to report.")
        import httpx
        models = httpx.get("http://localhost:11434/api/tags", timeout=10).json()["models"]
        model = next(m for m in models if m["name"] == "qwen2.5-coder:latest")
        manifest = dict(plan=plan, started_at=datetime.now(timezone.utc).isoformat(),
                        model=model, dataset_rows=len(frame),
                        dataset_counts=frame.intent.value_counts().to_dict(),
                        python=platform.python_version(), platform=platform.platform(),
                        logical_cpus=os.cpu_count(), packages={name: version(name) for name in
                        ["langchain-ollama", "langgraph", "sentence-transformers", "faiss-cpu", "scikit-learn", "matplotlib"]})
        atomic_json(manifest_path, manifest)
    pd.DataFrame(sampled).to_csv(output / "intent_sample.csv", index=False)
    if args.report_only:
        generate_reports(output)
        return
    checkpoints = Checkpoints(output, args.resume)
    supervisor = Supervisor()
    from agents.order_agent import OrderAgent
    from agents.refund_agent import RefundAgent
    with closing(sqlite3.connect(f"file:{database.DATABASE_PATH}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        orders = [dict(row) for row in connection.execute("SELECT * FROM orders ORDER BY order_id")]
    warmups = []

    def warm(name, function):
        start = time.perf_counter()
        try:
            result = function()
            error = result.get("error") if isinstance(result, dict) else None
        except Exception as exception:
            error = str(exception)
        warmups.append(dict(service=name, seconds=time.perf_counter() - start, error=error))
        atomic_json(output / "warmups" / f"{os.getpid()}.json", warmups)

    try:
        warm("Supervisor", lambda: supervisor.classify("How can I speak to customer support?"))
        for case in sampled:
            # Only question text crosses the model boundary, never labels/answers.
            checkpoints.run("intent", case, lambda c=case: supervisor.classify(c["question"]))
        generate_reports(output)  # Save usable partial metrics before agent evaluation.

        from agents.faq_agent import FAQAgent
        start = time.perf_counter()
        faq = FAQAgent()
        warmups.append(dict(service="FAQ initialization", seconds=time.perf_counter() - start, error=None))
        warm("FAQ", lambda: faq.answer("What is your return policy?"))
        faq_cases = FAQ_CASES if args.mode == "full" else FAQ_CASES[:3] + FAQ_CASES[-2:]
        for case in faq_cases:
            checkpoints.run("faq", case, lambda c=case: faq.answer(c["question"]))

        warm("Order", lambda: OrderAgent(customer_id=16).answer("Track order 1001"))
        cases = order_cases(orders)
        for case in cases if args.mode == "full" else cases[:2] + cases[-3:]:
            checkpoints.run("order", case, lambda c=case: OrderAgent(customer_id=c["customer_id"]).answer(c["question"]))

        # Isolate Python decisions from language extraction with controlled model
        # outputs, using an entirely temporary database. Live refund timing follows.
        with tempfile.TemporaryDirectory() as folder:
            db_path = Path(folder) / "evaluation.db"
            with closing(database.connect(db_path)) as connection:
                database.create_orders_table(connection)
                for i, (_, status, delivered, _, owner, _) in enumerate(REFUND_CASES, 9001):
                    connection.execute("INSERT INTO orders VALUES (?, ?, 'Evaluation product', 100, ?, '2026-01-01', ?, 'Card')",
                                       (i, owner, status, delivered))
                connection.commit()
            for i, (name, status, delivered, issue, owner, expected) in enumerate(REFUND_CASES, 9001):
                quote = {"damaged": "damaged", "incorrect": "incorrect", "condition_not_met": "I used the product"}.get(issue, "")
                question = f"Refund order {i}. {quote}".strip()
                case = dict(id=name, order_id=i, status=status, delivery_date=delivered,
                            issue=issue, owner=owner, customer_id=6, question=question,
                            expected_decision=expected, evaluation_date="2026-01-17")
                fake = Mock()
                fake.bind.return_value = fake
                fake.invoke.side_effect = [SimpleNamespace(content=json.dumps(dict(order_id=i, issue=issue, issue_quote=quote))),
                                           SimpleNamespace(content="Please use your verified response.")]
                agent = RefundAgent(customer_id=6, database_path=db_path, llm=fake)
                checkpoints.run("refund", case, lambda a=agent, q=question: a.answer(q, "2026-01-17"))

        warm("Refund", lambda: RefundAgent(customer_id=6).answer("Refund order 1002", "2026-10-10"))
        from workflow import build_workflow
        graph = build_workflow(faq_factory=lambda: faq)
        warm("Workflow", lambda: graph.invoke({"message": "What is your shipping policy?", "customer_id": 16}))
        for case in END_CASES if args.mode == "full" else END_CASES[:3] + END_CASES[13:15]:
            checkpoints.run("end_to_end", case, lambda c=case: graph.invoke(
                {"message": c["question"], "customer_id": c["customer_id"], "evaluation_date": c["evaluation_date"]}))

        for service, call in [
            ("Supervisor", lambda: supervisor.classify("Please track order 1001")),
            ("FAQ", lambda: faq.answer("When may I request a return?")),
            ("Order", lambda: OrderAgent(customer_id=16).answer("Track order 1001")),
            ("Refund", lambda: RefundAgent(customer_id=6).answer("Refund order 1002", "2026-10-10")),
        ]:
            for repetition in range(5 if args.mode == "full" else 1):
                checkpoints.run("performance", dict(id=f"{service}_{repetition}", service=service,
                                                     repetition=repetition), call)
    finally:
        preserved = protected_hashes() == protected
        atomic_json(output / "preservation.json", {"protected_files_unchanged": preserved})
        generate_reports(output)
        if not preserved:
            raise RuntimeError("A protected project file changed during evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = str(ROOT / ("evaluation" if args.mode == "full" else "evaluation/quick"))
    run(args)
