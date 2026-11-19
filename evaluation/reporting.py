"""Metrics/plots from saved observations. Never calls an LLM or changes agents."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from agents.supervisor import ROUTES


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def records(output, section):
    return [json.loads(path.read_text()) for path in sorted((output / "checkpoints" / section).glob("*.json"))]


def mean(values):
    return float(np.mean(values)) if values else None


def timings(values):
    return dict(n=len(values), mean_seconds=mean(values),
                median_seconds=float(np.median(values)) if values else None,
                p95_seconds=float(np.percentile(values, 95)) if values else None)


def class_metrics(truth, predicted, labels):
    p, r, f, support = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0)
    report = [dict(intent=label, precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(support[i]))
              for i, label in enumerate(labels)]
    return dict(n=len(truth), accuracy=float(accuracy_score(truth, predicted)),
                macro_precision=float(p.mean()), macro_recall=float(r.mean()), macro_f1=float(f.mean()),
                weighted_f1=float(np.average(f, weights=support))), report


def plot_matrix(matrix, rows, columns, title, path):
    fig, ax = plt.subplots(figsize=(max(8, len(columns) * .75), max(5, len(rows) * .55)))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(len(rows)):
        for j in range(len(columns)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "#173042", fontsize=8)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set(xlabel="Predicted", ylabel="Ground truth", title=title)
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def bar(labels, values, title, ylabel, path, unit=False):
    fig, ax = plt.subplots(figsize=(max(7, .8 * len(labels)), 5))
    rects = ax.bar(labels, values, color="#087f79")
    ax.bar_label(rects, fmt="%.3f", padding=3, fontsize=8)
    ax.set(title=title, ylabel=ylabel)
    ax.tick_params(axis="x", labelrotation=35)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    if unit:
        ax.set_ylim(0, 1.12)
    else:
        ax.set_ylim(0, max(values) * 1.25 if values else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def review_for(reviews, section, case_id, answer):
    entry = reviews.get(f"{section}/{case_id}")
    if entry is None:
        return None
    if entry.get("response_sha256") != hashlib.sha256(answer.encode()).hexdigest():
        raise ValueError(f"Stale review for {section}/{case_id}")
    if type(entry.get("semantic_correct")) is not bool or not entry.get("reviewer") or not entry.get("notes"):
        raise ValueError("Reviews require boolean semantic_correct, reviewer, and notes.")
    return entry


def generate_reports(output):
    from evaluate import LABELS, order_checks
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    manifest = json.loads((output / "manifest.json").read_text())
    reviews = json.loads((output / "reviews.json").read_text()) if (output / "reviews.json").exists() else {}
    summary = {}
    intent_rows = []
    for record in records(output, "intent"):
        case, result = record["case"], record["result"]
        prediction = "__error__" if record["error"] else result.get("intent", "__error__")
        if prediction not in ROUTES:
            prediction = "__error__"
        intent_rows.append(dict(**case, predicted_intent=prediction, error=record["error"], seconds=record["seconds"],
                                expected_agent=ROUTES[case["expected_intent"]],
                                predicted_agent=ROUTES.get(prediction, "__error__")))
    if intent_rows:
        frame = pd.DataFrame(intent_rows)
        frame.to_csv(output / "intent_predictions.csv", index=False)
        metrics, per_intent = class_metrics(frame.expected_intent, frame.predicted_intent, LABELS)
        metrics.update(target=manifest["plan"]["intent_target"], errors=int((frame.predicted_intent == "__error__").sum()),
                       unsupported_predictions=int((frame.predicted_intent == "unsupported").sum()))
        summary["intent"] = metrics
        write_json(output / "intent_metrics.json", metrics)
        pd.DataFrame(per_intent).to_csv(output / "classification_report.csv", index=False)
        labels = LABELS + ["unsupported", "__error__"]
        matrix = confusion_matrix(frame.expected_intent, frame.predicted_intent, labels=labels)[:len(LABELS), :]
        pd.DataFrame(matrix, index=LABELS, columns=labels).to_csv(output / "intent_confusion_matrix.csv")
        plot_matrix(matrix, LABELS, labels, "Supervisor intent classification", figures / "intent_confusion_matrix.png")
        shutil.copyfile(figures / "intent_confusion_matrix.png", output / "confusion_matrix.png")
        bar(LABELS, [row["f1"] for row in per_intent], "Per-intent F1 (fixed label set)", "F1", figures / "per_intent_f1.png", True)
        routing = []
        for agent, group in frame.groupby("expected_agent"):
            routing.append(dict(agent=agent, support=len(group), accuracy=float((group.expected_agent == group.predicted_agent).mean())))
        route_metrics = dict(n=len(frame), accuracy=float((frame.expected_agent == frame.predicted_agent).mean()),
                             per_agent=routing, definition="Per-agent accuracy is correct destinations / examples whose true destination is that agent.")
        summary["routing"] = route_metrics
        write_json(output / "routing_metrics.json", route_metrics)
        route_labels = sorted(set(ROUTES.values())) + ["__error__"]
        matrix = confusion_matrix(frame.expected_agent, frame.predicted_agent, labels=route_labels)
        pd.DataFrame(matrix, index=route_labels, columns=route_labels).to_csv(output / "routing_confusion_matrix.csv")
        plot_matrix(matrix, route_labels, route_labels, "Destination routing", figures / "routing_confusion_matrix.png")
        bar([r["agent"] for r in routing], [r["accuracy"] for r in routing], "Routing accuracy by ground-truth destination", "Accuracy", figures / "routing_accuracy.png", True)
        frame[frame.expected_intent != frame.predicted_intent].to_csv(output / "intent_errors.csv", index=False)

    faq_rows = []
    review_queue = []
    for rec in records(output, "faq"):
        case, result = rec["case"], rec["result"]
        answer = result.get("answer", "")
        sources = result.get("sources", [])[:3]
        relevant = set(sources) & set(case["sources"])
        hit = bool(relevant) if case["supported"] else None
        refusal = bool(re.search(r"could not verify|cannot verify|can't verify|couldn't verify", answer, re.I))
        auto = (all(re.search(p, answer, re.I) for p in case["patterns"]) if case["supported"] else refusal) and not rec["error"]
        review = review_for(reviews, "faq", case["id"], answer)
        row = dict(**case, answer=answer, retrieved_sources=sources, error=rec["error"], seconds=rec["seconds"],
                   hit_at_3=hit, source_precision_at_3=len(relevant)/len(sources) if sources and case["supported"] else (0.0 if case["supported"] else None),
                   automated_expected_fact_flags=bool(auto), automated_refusal_flag=refusal if not case["supported"] else None,
                   review=review)
        faq_rows.append(row)
        review_queue.append(dict(section="faq", id=case["id"], question=case["question"], expected=case["expected"],
                                 answer=answer, response_sha256=hashlib.sha256(answer.encode()).hexdigest()))
    if faq_rows:
        write_json(output / "faq_results.json", faq_rows)
        supported = [r for r in faq_rows if r["supported"]]
        unsupported = [r for r in faq_rows if not r["supported"]]
        reviewed = [r for r in faq_rows if r["review"]]
        reviewed_unsupported = [r for r in unsupported if r["review"]]
        summary["faq"] = dict(n=len(faq_rows), supported_n=len(supported), unsupported_n=len(unsupported),
                              retrieval_hit_at_3=mean([r["hit_at_3"] for r in supported]),
                              source_precision_at_3=mean([r["source_precision_at_3"] for r in supported]),
                              automated_expected_fact_flag_rate=mean([r["automated_expected_fact_flags"] for r in faq_rows]),
                              automated_refusal_flag_rate=mean([r["automated_refusal_flag"] for r in unsupported]),
                              reviewed_n=len(reviewed), semantic_correct_on_reviewed=mean([r["review"]["semantic_correct"] for r in reviewed]),
                              reviewed_unsupported_n=len(reviewed_unsupported),
                              refusal_correct_on_reviewed=mean([r["review"]["semantic_correct"] for r in reviewed_unsupported]))

    order_rows = []
    for rec in records(output, "order"):
        order_rows.append(dict(**rec, checks=order_checks(rec["case"], rec["result"])))
    if order_rows:
        write_json(output / "order_results.json", order_rows)
        summary["order"] = dict(n=len(order_rows), success_rate=mean([r["checks"]["success"] for r in order_rows]),
                                record_accuracy=mean([r["checks"]["records_correct"] for r in order_rows]),
                                no_fabricated_date_rate=mean([r["checks"]["no_fabricated_dates"] for r in order_rows]),
                                errors=sum(bool(r["error"]) for r in order_rows))

    refund_rows = []
    for rec in records(output, "refund"):
        result = rec["result"]
        correct = result.get("eligibility_status") == rec["case"]["expected_decision"] and not rec["error"]
        privacy = rec["case"]["owner"] == 6 or (result.get("order_id") is None and result.get("answer") == "That order is unavailable for your account. Please check the order ID.")
        refund_rows.append(dict(**rec, decision_correct=bool(correct), privacy_correct=privacy))
    if refund_rows:
        write_json(output / "refund_results.json", refund_rows)
        summary["refund"] = dict(n=len(refund_rows), decision_accuracy=mean([r["decision_correct"] for r in refund_rows]),
                                 privacy_pass=all(r["privacy_correct"] for r in refund_rows), model_extraction="mocked to isolate deterministic policy logic")
        labels = ["eligible", "ineligible", "manual_review", "unknown", "__error__"]
        matrix = confusion_matrix([r["case"]["expected_decision"] for r in refund_rows],
                                  [r["result"].get("eligibility_status", "__error__") if not r["error"] else "__error__" for r in refund_rows], labels=labels)
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output / "refund_confusion_matrix.csv")

    e2e_rows = []
    for rec in records(output, "end_to_end"):
        case, result = rec["case"], rec["result"]
        answer = result.get("response", "")
        route = result.get("selected_agent") == ROUTES[case["expected_intent"]]
        auto = all(re.search(pattern, answer, re.I) for pattern in case["patterns"]) and not rec["error"]
        agent = result.get("agent_result", {})
        if case["decision"]:
            auto = auto and agent.get("eligibility_status") == case["decision"]
        privacy = None
        if case["privacy"]:
            privacy = (answer == "That order is unavailable for your account. Please check the order ID."
                       and not agent.get("orders") and agent.get("order_id") is None)
            auto = auto and privacy
        review = review_for(reviews, "end_to_end", case["id"], answer)
        e2e_rows.append(dict(**rec, route_correct=route, automated_response_flags=bool(auto), privacy_pass=privacy,
                             review=review, task_success=bool(route and auto and review["semantic_correct"]) if review else None))
        review_queue.append(dict(section="end_to_end", id=case["id"], question=case["question"], expected=case["expected"],
                                 answer=answer, response_sha256=hashlib.sha256(answer.encode()).hexdigest()))
    if e2e_rows:
        write_json(output / "end_to_end_results.json", e2e_rows)
        reviewed = [r for r in e2e_rows if r["review"]]
        summary["end_to_end"] = dict(n=len(e2e_rows), route_accuracy=mean([r["route_correct"] for r in e2e_rows]),
                                     automated_task_flag_rate=mean([r["route_correct"] and r["automated_response_flags"] for r in e2e_rows]),
                                     reviewed_n=len(reviewed), task_success_rate=mean([r["task_success"] for r in reviewed]) if len(reviewed) == len(e2e_rows) else None,
                                     privacy_n=sum(r["privacy_pass"] is not None for r in e2e_rows),
                                     strict_unavailable_response_rate=mean([r["privacy_pass"] for r in e2e_rows if r["privacy_pass"] is not None]),
                                     privacy_reviewed_n=sum(r["case"]["privacy"] and isinstance((r["review"] or {}).get("privacy_preserved"), bool) for r in e2e_rows),
                                     privacy_preserved_on_reviewed=mean([r["review"]["privacy_preserved"] for r in e2e_rows if r["case"]["privacy"] and isinstance((r["review"] or {}).get("privacy_preserved"), bool)]),
                                     latency=timings([r["seconds"] for r in e2e_rows]))
    write_json(output / "review_queue.json", review_queue)

    performance = records(output, "performance")
    if performance:
        latency = {name: dict(**timings([r["seconds"] for r in performance if r["case"]["service"] == name]),
                             errors=sum(bool(r["error"]) for r in performance if r["case"]["service"] == name))
                   for name in sorted({r["case"]["service"] for r in performance})}
        summary["performance"] = latency
        write_json(output / "latency_metrics.json", latency)
        write_json(output / "latency_observations.json", performance)
        bar(list(latency), [r["mean_seconds"] for r in latency.values()], "Warm local service latency (mean)", "Seconds", figures / "agent_latency.png")
    write_json(output / "metrics.json", summary)
    write_report(output, manifest, summary, intent_rows, faq_rows, e2e_rows, reviews)


def write_report(output, manifest, summary, intents, faq, e2e, reviews):
    def pct(value):
        return "not yet measured/reviewed" if value is None else f"{value:.2%}"
    lines = ["# ShopAssist AI — Final Evaluation", "",
             "## Project and datasets", "",
             "ShopAssist combines local Qwen classification, LangGraph routing, policy RAG, SQLite order lookup, deterministic refund assessment, and a Streamlit UI. This evaluates the existing system without training or changing its agents.", "",
             f"Dataset: {manifest['dataset_rows']:,} Bitext-derived labeled queries, ten intents; 200 synthetic existing orders; four fictional policy documents. Dataset responses are NOT used as answer truth for DemoCart policies.", "",
             "## Methodology and reproducibility", "",
             f"Mode: **{manifest['plan']['mode']}**; seed: **{manifest['plan']['seed']}**; target labeled sample: **{manifest['plan']['intent_target']}**. Equal-size random sampling without replacement within each original intent, then seeded shuffling. Original labels, wording, typos, and placeholders are preserved. Only question text is supplied to Supervisor.classify().", "",
             "Failed/invalid predictions use __error__ and remain in the denominator. Valid unsupported predictions also remain errors against the ten dataset labels. Macro scores average the fixed ten ground-truth labels; weighted F1 uses their sample support; undefined precision is zero. Confusion matrices retain unsupported/error columns. Routing uses the production intent-to-destination mapping; per-agent accuracy is conditional on the true destination (recall-like), not precision.", "",
             "All checkpoints are atomic per-case JSON files. Manifest hashes freeze data, policies, application code, and evaluation cases. --resume reuses successes AND failures rather than selectively retrying mistakes. --report-only regenerates artifacts without inference. Models can still be nondeterministic across hardware/runs.", "",
             "FAQ set: 24 independently authored cases (18 supported, 6 unsupported), including paraphrases and multi-policy prompts. Hit@3 means any predeclared relevant source in the first three; source relevance is filename precision@3, not a semantic ranking judgment. Unsupported questions have no relevant source and are excluded from retrieval denominators.", "",
             "Rubric for qualitative review: (1) required facts/conditions are included; (2) every asserted policy or order fact is supported by the frozen policy/record; (3) no invented dates, status, approvals, transfers, or real escalation; (4) unsupported questions explicitly abstain without asserting an unavailable policy. A substantive omission, contradiction, or invented rule fails semantic_correct. Refusal correctness uses this same rubric for unsupported questions. Automated regex flags are screening aids, NOT proof of semantic correctness.", "",
             "Order checks compare full structured records and complete approved factual text against independently serialized database fields; dates must be a subset of recorded dates. Refund fixtures use a temporary database and mocked extraction/generation to isolate policy decisions; expected classes were explicitly specified in advance, not obtained from the tested function. eligible means conditional request eligibility, not approval.", "",
             "End-to-end: 20 independent single-request scenarios, including all ten intents, fallback, privacy and unsupported operations. Success requires correct destination, automated response/decision checks and qualitative response review. Without complete review coverage the final task-success rate is withheld. No conversational memory or multi-turn capability is claimed.", "",
             "## Intent classification", ""]
    if "intent" in summary:
        m = summary["intent"]
        lines += [f"Evaluated **{m['n']}/{m['target']}** queries; invalid/failed: **{m['errors']}**; valid unsupported predictions: **{m['unsupported_predictions']}**.", "",
                  "| Metric | Value |", "|---|---:|"]
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
            lines.append(f"| {key} | {pct(m[key])} |")
        lines += ["", "![Intent matrix](figures/intent_confusion_matrix.png)", "", "![Per-intent F1](figures/per_intent_f1.png)"]
    lines += ["", "## Routing and confusion analysis", ""]
    if "routing" in summary:
        m = summary["routing"]
        lines += [f"Destination accuracy: **{pct(m['accuracy'])}**. This is separate from intent accuracy: confusing two intents routed to the same specialist can still select the correct destination.", "",
                  "| Ground-truth destination | Support | Correct destination rate |", "|---|---:|---:|"]
        for row in m["per_agent"]:
            lines.append(f"| {row['agent']} | {row['support']} | {pct(row['accuracy'])} |")
        pairs = Counter((r["expected_intent"], r["predicted_intent"]) for r in intents if r["expected_intent"] != r["predicted_intent"])
        lines += ["", "Most common intent confusions:"]
        lines += [f"- {a} → {b}: {count}" for (a, b), count in pairs.most_common(10)] or ["- None observed in this sample."]
        lines += ["", "![Routing accuracy](figures/routing_accuracy.png)"]
    lines += ["", "## Agent-level results", "", "```json", json.dumps({k: summary[k] for k in ["faq", "order", "refund"] if k in summary}, indent=2), "```", "",
              "### Review provenance", ""]
    if reviews:
        reviewers = Counter(entry.get("reviewer", "unspecified") for entry in reviews.values())
        lines += [f"- {reviewer}: {count} responses." for reviewer, count in reviewers.items()]
        lines += ["Reviews are saved with response hashes and case-specific notes in reviews.json. Coding-assistant qualitative review is not independent human review. No human-reviewed correctness is claimed unless explicitly identified as such."]
        disagreements = [r["id"] for r in faq if r["review"] and r["automated_expected_fact_flags"] != r["review"]["semantic_correct"]]
        if disagreements:
            lines += ["", "FAQ keyword flags and qualitative judgments disagree on: " + ", ".join(disagreements) + ". Equal aggregate scores, if present, do not imply agreement on individual responses."]
    else:
        lines += ["No qualitative reviews have been supplied. Semantic correctness is unmeasured; automated flags must not be described as semantic accuracy."]
    lines += ["", "## End-to-end results", "", "```json", json.dumps(summary.get("end_to_end", {}), indent=2), "```", "",
              "The strict unavailable-response check requires the exact generic account-unavailable message and no returned records. A safe fallback can fail this handling check without disclosing information. Privacy-preservation review separately checks the answer and structured result for disclosure; it does not waive routing or task failures.", "",
              "## Latency", "",
              "Durations use time.perf_counter around actual invocations. One warm-up per service per process is excluded and separately recorded under warmups/. FAQ initialization is also recorded separately. Five repeated representative calls per service in full mode are descriptive, not a load benchmark. End-to-end timings include routing plus the selected node; no-op/escalation routes are cheaper. Errors are retained in latency statistics. Percentile p95 uses NumPy's linear interpolation. No parallel calls were launched by this evaluator.", "",
              "```json", json.dumps(summary.get("performance", {}), indent=2), "```"]
    if "performance" in summary:
        lines += ["", "![Service latency](figures/agent_latency.png)"]
    lines += ["", "## Error analysis", ""]
    failures = [r for r in faq if r.get("review") and not r["review"]["semantic_correct"]]
    for row in failures:
        lines.append(f"- FAQ {row['id']}: {row['review']['notes']}")
    for row in e2e:
        if not row["route_correct"] or not row["automated_response_flags"] or (row["review"] and not row["review"]["semantic_correct"]):
            lines.append(f"- E2E {row['case']['id']}: expected {row['case']['expected_intent']}, obtained {row['result'].get('intent')}; "
                         f"automated checks={row['automated_response_flags']}; review={row['review']['notes'] if row['review'] else 'pending'}.")
    lines += ["See intent_errors.csv for every intent error, including original queries. These are observations against dataset labels, not relabeling decisions. No production prompts or rules were tuned on the evaluation sample.", "",
              "## Limitations", "",
              "- Balanced 200-query sample is not a full 9,975-row evaluation. The dataset is synthetic/template-heavy and can contain ambiguous labels, placeholders, and near-duplicates; rows are sampled without replacement but template families are not held out.",
              "- Existing prompts and earlier smoke tests predate this evaluation; this is not a pristine blinded research benchmark. No fine-tuning occurs here.",
              "- Only four policy documents exist; retrieving three gives a generous Hit@3 baseline. Filename relevance is not complete semantic evidence. Qualitative review by the coding assistant has reviewer bias and needs independent human confirmation.",
              "- Refund deterministic accuracy excludes natural-language extraction errors. Real refund execution is absent. Live refund latency and workflow scenarios exercise the local model separately.",
              "- Safety cases are small and do not prove universal resistance to prompt injection. Authentication is simulated. No real refunds/cancellations or human tickets are created.",
              "- Latencies reflect this machine, model/cache state and local load; five repetitions cannot characterize production performance or concurrent users.", "",
              "## Conclusion", "",
              "Measured counts and results above describe this frozen evaluation only. Inspect error records and review provenance before using the scores as project claims. Unmeasured sections and incomplete review coverage remain explicit rather than being treated as passing.", "",
              "Metric reference: [scikit-learn precision/recall/F-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)."]
    lines += ["", "## Saved artifacts and environment", "",
              "Per-intent precision/recall/F1/support: [classification_report.csv](classification_report.csv). Raw predictions: [intent_predictions.csv](intent_predictions.csv). Cases and responses: faq_results.json, order_results.json, refund_results.json, end_to_end_results.json. Qualitative review: [reviews.json](reviews.json). All per-case observations, including failures, are under checkpoints/.", "",
              f"Local model: `{manifest['model']['name']}`; digest: `{manifest['model']['digest']}`. Python: {manifest['python']}; platform: {manifest['platform']}; logical CPUs: {manifest['logical_cpus']}. Exact library versions and protected-file hashes are in [manifest.json](manifest.json). This records the environment but does not establish GPU utilization or isolate other machine load.", "",
              "The preservation check is saved in [preservation.json](preservation.json). Reproduction and review instructions: [README.md](README.md)."]
    (output / "FINAL_EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
