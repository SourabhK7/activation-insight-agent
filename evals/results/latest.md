# Latest evaluation results

Generated: 2026-07-15 09:21 UTC
Runs per arm: 10 (seed base: 1000)
Diagnosis model: claude-sonnet-5 · Judge model: claude-sonnet-5

## Aggregate scores (0–12 per run)

| Arm | N (scored) | N (judge failed) | Mean total | Stdev |
|---|---:|---:|---:|---:|
| structured | 9 | 1 | 7.33 | 0.87 |
| naive | 4 | 6 | 11.75 | 0.50 |

## Per-criterion mean scores (0–2)

| Arm | c1 mobile→payment | c2 paid-social→cart | c3 CX region | c4 no-fabrication | c5 numerical | c6 calibrated |
|---|---:|---:|---:|---:|---:|---:|
| structured | 0.00 | 1.56 | 0.22 | 1.56 | **2.00** | 2.00 |
| naive | 2.00 | 2.00 | 2.00 | 1.75 | **2.00** | 2.00 |

## What the numbers actually say

Two findings, neither of them the one the design assumption predicted.

**1. The numerical-accuracy claim is not supported by this run.**
Criterion 5 was the direct test of the design thesis — the specific worry that the naive baseline would silently produce wrong rates because "LLMs are unreliable at arithmetic." Both arms scored 2.00 (perfect) on numerical accuracy across every successful run. On this task at this scale, Sonnet 5 handled the arithmetic. The narrow claim does not hold up under measurement.

**2. The naive baseline scored higher on total.**
When the judge succeeded, naive scored 11.75/12 vs structured's 7.33/12. The naive prompt produces longer, more thorough diagnoses that surface the mobile-payment pattern (c1), the CX region (c3), and the paid-social issue (c2) more consistently. The structured arm systematically buries the mobile-payment pattern behind a "desktop advantage at shipping" framing — a real bug in how the pipeline presents divergent segments (mobile and desktop are the same information flipped, and the current cohort ranking prefers the direction with the smaller-magnitude but larger-N segment).

## Two large caveats before drawing conclusions

**N is small and unequal.** Structured has 9 scored runs; naive has 4. The naive comparison rests on the assumption that the 6 judge failures are unbiased with respect to diagnosis quality — plausible (they look like judge-token exhaustion on the longer naive outputs), but not verified. Any total-score comparison across arms at these N's is directional at best.

**One judge, one model family.** Both the arm producing the diagnosis and the judge scoring it are Sonnet 5. LLM-as-judge is not neutral; a judge that shares training data with the arm being scored has systematic blind spots. See `../rubric.md` for the fuller limitations list.

## What this means for the repo

The structured design is still worth keeping — it's more predictable, cheaper (shorter prompts, less reasoning), and the diagnosis surface area is smaller — but the pitch "we do this because LLM arithmetic is unreliable" needs revision on the strength of this evidence. A more honest framing: the structured pipeline gives determinism and inspectability for the numbers, at some cost in diagnostic completeness. Both matter; neither dominates.

The c1 miss (mobile-payment framing) is a real actionable bug and is the next thing worth fixing.

## Raw runs

`results/raw_runs.jsonl` — one JSON line per (arm, seed) run with the full diagnosis text, judge scorecard, per-criterion reason strings, token counts, and duration. Every score in the table above is auditable there.
