"""Live smoke tests and modest rule-based evaluation, not a semantic proof."""

import json
from pathlib import Path
import re

from agents.faq_agent import FAQAgent

# Handwritten facts: every pattern in a case must appear in the answer.
CASES = [
    ("What is your refund policy?", "refund_policy.md",
     [r"5\s*[-–—]\s*7\s+business days", r"approv"]),
    ("How long does delivery take?", "shipping_policy.md",
     [r"3\s*[-–—]\s*7\s+business days", r"dispatch"]),
    ("Can I cancel an order after it has shipped?", "cancellation_policy.md",
     [r"cannot|can't|can’t|not (?:be |directly )?cancel", r"shipped|shipment",
      r"Processing", r"return", r"delivery|delivered"]),
    ("How many days do I have to return a product?", "return_policy.md",
     [r"7 days", r"delivery|delivered"]),
    ("What should I do if I receive a damaged product?", "return_policy.md",
     [r"support", r"manual review", r"photo"]),
    ("Do you offer international shipping?", "shipping_policy.md",
     [r"could not verify|cannot verify|can't verify|couldn't verify"]),
]


def main():
    try:
        agent = FAQAgent()
    except Exception as error:
        print(f"FAQ setup failed: {error}")
        return 1
    results = []
    for question, expected_source, patterns in CASES:
        result = agent.answer(question)
        retrieval_ok = expected_source in result["sources"]
        facts_ok = not result["error"] and all(
            re.search(pattern, result["answer"], flags=re.IGNORECASE)
            for pattern in patterns
        )
        # For international shipping, also reject an invented yes/no commitment.
        if "international" in question.lower():
            facts_ok = facts_ok and not re.search(
                r"we (?:do |don't |do not )?(?:offer|ship)|yes[,!]|worldwide",
                result["answer"], flags=re.IGNORECASE,
            )
        print(f"\nCustomer: {question}")
        print(f"Sources: {', '.join(result['sources'])}")
        print(f"Answer: {result['answer']}")
        if result["error"]:
            print(f"Service error: {result['error']}")
        print(f"Retrieval: {'PASS' if retrieval_ok else 'FAIL'}; "
              f"Expected facts: {'PASS' if facts_ok else 'FAIL'}")
        results.append(dict(question=question, **result,
                            retrieval_pass=retrieval_ok, facts_pass=bool(facts_ok)))
    report = Path(__file__).resolve().parent / "data" / "faq_evaluation.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    retrieval_count = sum(item["retrieval_pass"] for item in results)
    fact_count = sum(item["facts_pass"] for item in results)
    print(f"\nRetrieved expected document: {retrieval_count}/{len(CASES)}")
    print(f"Expected-fact checks: {fact_count}/{len(CASES)}")
    print("These keyword checks do not establish full correctness or grounding.")
    print(f"Saved answers and evaluation: {report}")
    return 0 if retrieval_count == fact_count == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
