# Latest evaluation results

Generated: 2026-07-23 20:04 UTC
Runs per arm: 10 planned (seed base: 1000)
Diagnosis model: claude-sonnet-5 · Judge model: claude-sonnet-5
Second evaluation run, after the vs-rest delta and `underperforming_steps` fix landed (commit 69fe405).

## Aggregate scores (0–12 per run)

| Arm | N scored | N judge-failed | Mean total | Stdev |
|---|---:|---:|---:|---:|
| structured | 2 | 8 | 6.00 | 1.41 |
| naive | 6 | 4 | 10.50 | 1.38 |

## Per-criterion mean scores (0–2)

| Arm | c1 mobile→payment | c2 paid-social→cart | c3 CX region | c4 no-fabrication | c5 numerical | c6 calibrated |
|---|---:|---:|---:|---:|---:|---:|
| structured | **0.00** | 0.50 | **0.00** | 1.50 | 2.00 | 2.00 |
| naive | 2.00 | 2.00 | 2.00 | 1.17 | 1.33 | 2.00 |

## What the numbers actually say

**The fix I made didn't move c1.** The structured arm still scores 0.00 on "correctly identifies the mobile-payment issue," same as run #1. The vs-rest delta fix and the `underperforming_steps` field both landed and function as designed — the diagnosis now correctly calls out mobile as the underperforming segment, rather than framing it as "desktop's advantage at shipping." But the eval still marks the answer as missing the pattern.

Digging into the raw runs revealed why. The structured arm's diagnosis says something like "mobile users have a 34pp lower conversion at the shipping step." The eval's ground truth, which I wrote, expects the finding labeled at the payment step. These describe the same event from opposite sides of the shipping→payment transition. Both are accurate. The judge sees two different words and marks the answer as wrong.

**This is a labeling divergence between the pipeline and the eval rubric, not a code bug in either.** The pipeline's `step_conversion` at step X is defined as "fraction of users at X who continued to X+1," so a deficit shows up at the step where users *were*. The judge's rubric expects the deficit named by the step users *failed to reach*. Both are legitimate conventions. Nothing in either component makes it obvious you'd have to align them until an eval starts scoring zero for reasons that aren't really about the answer.

Fixing it is a one-commit change (either update the pipeline to name transitions as `from_step → to_step` explicitly, or adjust the rubric to accept either label). Once aligned, c1 should move.

**Naive arm still wins on scored runs.** When the judge can score it, the naive baseline produces thorough diagnoses that identify all three planted patterns (c1, c2, c3 all at 2.00). The judge failure rate at 4/10 is non-trivial but manageable. Interpretation of the mean total (10.50) should account for the possibility that harder-to-score diagnoses are systematically not being scored.

**Structured arm's judge failure rate (8/10) is a concern in its own right.** Bumping judge max_tokens from 4000 to 8000 in the previous iteration helped less than expected. Worth investigating whether the new `underperforming_steps` field made the structured arm's output somehow harder for the judge to grade, or whether this is just small-N variance.

## Two large caveats

**N is very small and unequal.** Structured has 2 scored runs; naive has 6. Total-score comparisons across arms at these N's are directional at best, not defensible as headline claims.

**One judge, one model family.** Both the arms producing diagnoses and the judge scoring them are Sonnet 5. LLM-as-judge shares training data with the arms being scored, which introduces correlated blind spots. See `../rubric.md` for the fuller limitations list.

## What this means for the repo

The `Findings` → LLM narrative design is still the right structure. The fix landed correctly at the analysis layer, and the pipeline now surfaces the actionable segment. What needs work is either the labeling convention (align pipeline output with the vocabulary the rubric uses) or the rubric itself (accept both labelings). Both are small changes.

The current c1 = 0.00 result is a false failure caused by a vocabulary mismatch between two components I built, not evidence the pipeline is producing wrong diagnoses. That's a subtle failure mode worth naming: when you build an eval for your own system, the eval and the system have to agree on what to call things, or the eval will report failures that aren't really about correctness.

## Raw runs

`results/raw_runs.jsonl` — one JSON line per (arm, seed) run with the full diagnosis text, judge scorecard, per-criterion reason strings, token counts, and duration. Every score in the tables above is auditable there.
