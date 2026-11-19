# Refund Agent

This agent assesses **new return/refund requests** using the existing SQLite
order helpers and the fictional DemoCart policies. It never approves a refund
or updates orders/payments.

```bash
python -m agents.refund_agent --customer-id 6 "Can I get a refund for order 1002?"
python -m agents.refund_agent --customer-id 6 --evaluation-date 2026-10-10 "Can I get a refund for order 1002?"
python test_refund_agent.py
python test_refund_agent.py --live
```

The CLI customer ID simulates a trusted session. An application must supply the
authenticated customer's ID separately from their message:

```python
from agents.refund_agent import RefundAgent

agent = RefundAgent(customer_id=session_customer_id)
result = agent.answer(customer_message, evaluation_date="2026-11-10")
```

`evaluation_date` accepts a Python date or ISO date string. If omitted, it uses
the current date in Asia/Kolkata. The result contains `order_id`,
`eligibility_status`, `reason`, `policy_sources`, `answer`, `evaluation_date`,
`error`, and `used_fallback`. Treat `error` as an internal diagnostic.
Unavailable/foreign orders both return the same message and a null order ID.

Policy interpretations and limits identified before implementation:

- `eligible` means **conditionally eligible to request an ordinary return**, not
  verified item eligibility or refund approval. SQLite stores no condition,
  packaging, accessories, return-request history, inspection, or refund records.
  The answer states the unused-item/packaging/accessory and inspection conditions.
  An explicit report of unmet ordinary item conditions is assessed as ineligible
  based on that report; it is not an independently verified inspection.
- The seven-day window is implemented as elapsed calendar dates, inclusive:
  `0 <= evaluation_date - delivery_date <= 7 days`. The policies do not specify
  time-of-day cutoffs. Exactly seven days is within the request window.
- Outside the window, `ineligible` concerns a **new standard request**. An earlier
  on-time request cannot be checked; the response refers that question to support.
- Damaged/incorrect products require `manual_review`, even outside the ordinary
  window. The policies do not specify a separate deadline or guaranteed outcome.
- Processing orders are referred to cancellation. Cancelled orders require
  support verification because no cancellation-refund terms/status data exist.
  Shipped, delayed, missing/invalid/future delivery dates produce `unknown`.
- No payment date is calculated from the 5–7 business-day processing statement:
  no refund approval date is available.

Python determines the result using manually reviewed policy rules. Policy-file
SHA-256 fingerprints prevent stale decisions after files change. Review the rules
and update fingerprints together when deliberately updating company policy. Missing
or changed policies result in `unknown`; the agent never edits them.

Qwen extracts an explicit order ID and product-issue category. The ID must occur
in the message; reported issues need an exact quotation. This is language
interpretation, not proof of the customer's claim. Common damage/incorrect-item
keywords conservatively force manual review even if Qwen misses them, including
some negated mentions. Other wording can still be misclassified by the model.
Only the trusted session ID is used for the existing parameterized order lookup.

For explanation, Qwen chooses between approved responses containing the Python
decision. Exact validation rejects added approvals, dates, or transactions and
falls back to deterministic text. If extraction fails, no order lookup occurs;
if explanation fails, the verified decision remains available.

The default test script runs 21 deterministic tests with a temporary database
and mocked model outputs. `--live` additionally runs five actual Qwen cases on
existing orders, including privacy and damage escalation. These tests cover the
specified cases, not all possible natural-language phrasing.
