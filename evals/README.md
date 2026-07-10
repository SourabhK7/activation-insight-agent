# Evaluation harness

An A/B evaluation of the core design decision in this repo: *does moving arithmetic out of the LLM's hands actually improve the quality of the diagnosis?*

## What this measures

Two arms, same synthetic data, same downstream rubric:

**Arm A — structured (the design under test):**
Raw events → pandas computes findings → structured findings JSON → Claude writes diagnosis.

**Arm B — naive baseline (fair comparison, not a strawman):**
Raw events → pandas aggregates step counts + per-segment counts → Claude is asked to compute conversion rates and diagnose.

Both arms see the same information. The only difference is *who computes the rates* — Python or the LLM. That's the design decision under test.

Both diagnoses are scored by an LLM judge (Claude) against a 6-criterion rubric ([rubric.md](rubric.md)). The judge is shown the ground truth for the specific synthetic dataset each run was on, so criterion 5 (numerical accuracy) is graded against an oracle, not against the judge's own arithmetic.

## Why this exists

The README claims:

> *"By doing all the math in pandas and sending only the structured findings to Claude, we avoid the whole class of 'the LLM confidently computed the wrong conversion rate' bugs."*

That's a testable claim. This harness tests it.

If Arm A meaningfully outperforms Arm B — especially on criterion 5 — the design assumption holds up under actual measurement. If the arms score the same, the design decision doesn't matter and the code could be simplified. Either way, the answer is worth having.

## How to run it

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python evals/run_eval.py --n 10
```

Options:

- `--n 10` — runs per arm (default 10). Each run uses a fresh seed, so the arms are graded across data variants, not on a single lucky dataset.
- `--n-users 20000` — users per synthetic dataset. Smaller = faster + cheaper. Default is deliberately smaller than the 50k in the demo, because 10 runs × 2 arms adds up.
- `--diagnosis-model claude-sonnet-5` — the model producing the diagnoses.
- `--judge-model claude-sonnet-5` — the model grading them. Can be the same as the diagnosis model; using different models is more defensible but more expensive.
- `--arms structured,naive` — comma-separated arms to run. Useful for re-running just one arm.
- `--reset` — delete `results/raw_runs.jsonl` before starting.

Cost: for `--n 10 --n-users 20000` on Sonnet 5, expect roughly $1 total. Runs take ~4–8 minutes depending on API latency.

## Output

- **`results/latest.md`** — human-readable summary: mean total per arm, stdev, per-criterion means.
- **`results/raw_runs.jsonl`** — one line per (arm, seed) run. Contains the full diagnosis text, judge scorecard, and per-criterion reasons. Every score is auditable — if a specific run's score looks wrong, open the JSONL and inspect it.

## Interpreting the results

Look at criterion 5 first. That's the direct test of the design thesis. A large gap on criterion 5 with a small or zero gap elsewhere would be the cleanest possible confirmation: "the structured design specifically improves numerical accuracy without hurting anything else."

Look at variance too. With N=10 the confidence interval on the mean is wide. Treat differences of less than ~1 point on total (out of 12) as noise.

Criterion 4 (no fabrication) is the second most interesting: if Arm B, forced to compute rates itself, also invents patterns to explain the numbers it computed, that's a compounding failure mode.

## Limitations

See [rubric.md](rubric.md#limitations-of-llm-as-judge) for the honest list. Summary:

1. LLM-as-judge is not bulletproof. The judge shares training data with the arms.
2. N=10 is small. This is a directional check, not a benchmark.
3. Ground truth for criterion 5 is derived from the same pandas pipeline that powers Arm A. Arm A therefore has a subtle unfair advantage on any judgment where the "correct" number matches the pipeline's exact rounding. This is why criterion 5 uses a 1pp tolerance — big enough that rounding doesn't decide the outcome.
4. The naive baseline is *fair but not maximally sophisticated*. A real DS handed the aggregated counts would compute the rates in a notebook. We're comparing "aggregate counts + LLM computes" to "structured findings + LLM writes prose," which is exactly the design tradeoff in the repo — but doesn't cover "aggregate counts + LLM writes code + LLM executes it," which is a different agent design.

Nothing here is meant as a general claim about LLM arithmetic. It's a claim about *this specific pipeline*, tested with a rubric designed for this specific task.
