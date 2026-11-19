# Reproduce the final evaluation

Run from the project root with Ollama serving `qwen2.5-coder:latest`:

```bash
python -m unittest -v test_evaluate.py
HF_HUB_OFFLINE=1 python evaluate.py --mode quick --seed 42
HF_HUB_OFFLINE=1 python evaluate.py --mode full --seed 42
HF_HUB_OFFLINE=1 python evaluate.py --mode full --seed 42 --resume
python evaluate.py --mode full --seed 42 --report-only
```

Quick mode writes to `evaluation/quick/`; full mode to `evaluation/`. A different
`--output` creates an independent run. Full mode samples exactly 20 queries per
original intent (200 total). Quick mode samples two per intent and smaller live
agent subsets. Both run the complete deterministic refund fixture set.

Each case is atomically checkpointed before continuing. Ctrl+C preserves all
completed cases and generates partial reports in `finally`. Resume reuses failures
as well as successes; use a new directory for a genuinely fresh repeat. A changed
seed, mode, input, production implementation, case specification or runner is
rejected by the manifest comparison. Run only one evaluator per output directory.
Do not edit production code during measurement. No data or production code is
written by this evaluator; refund fixtures are created in a temporary directory.

Additional evaluation dependencies already available in this environment:
`matplotlib==3.10.8` and `scikit-learn==1.8.0`. Existing project requirements supply
the agent dependencies. Offline mode needs the existing MiniLM cache.

`intent_sample.csv` records original zero-based CSV data-row indexes and labels.
Only query text goes to the existing Supervisor. The manifest records protected
file hashes, model digest, versions and machine information. Service warm-ups and
initialization timings are separate from scored observations. Full performance
measurement uses five calls per service; end-to-end uses 20 independent scenarios.

Semantic review is deliberately separate from keyword screening. Review
`review_queue.json` against frozen policy files and expected facts in `cases.py`.
Save entries in `reviews.json`, keyed by `faq/<case_id>` or
`end_to_end/<case_id>`, with these fields:

```json
{
  "faq/faq_01": {
    "response_sha256": "copy the exact hash from review_queue.json",
    "reviewer": "identify the actual reviewer; distinguish human from AI review",
    "semantic_correct": true,
    "notes": "Explain completeness, grounding, and any policy omissions."
  }
}
```

Review rubric: all required facts/conditions, no unsupported claims, no invented
actions or dates, and explicit abstention when the requested policy is absent.
Substantive omissions count as failures. Hashes bind review to actual output.
For privacy scenarios, also record a boolean `privacy_preserved` after inspecting
both the response and the checkpoint's structured result. This distinguishes
safe fallback refusals from successful handling with the exact expected generic
unavailable response; safe fallback does not override a task or routing failure.
Regenerate reports with `--report-only`; this performs no model calls. A coding
assistant's review is not independent human assessment. No human-review scores are
claimed unless a human actually completes this step.

See `FINAL_EVALUATION.md` for measured metrics and limitations. Raw failures remain
in checkpoints, prediction tables, and `intent_errors.csv`. Routing correctness
is evaluated separately because several intents share the same destination.
