# Latest evaluation results

Generated: 2026-07-23 20:04 UTC
Runs per arm: 10 planned (seed base: 1000)
Diagnosis model: claude-sonnet-5 · Judge model: claude-sonnet-5
This is the second run, after the vs-rest delta and `underperforming_steps` fix (commit 69fe405).

## Aggregate scores (0 to 12 per run)

| Arm | N scored | N judge-failed | Mean total | Stdev |
|---|---:|---:|---:|---:|
| structured | 2 | 8 | 6.00 | 1.41 |
| naive | 6 | 4 | 10.50 | 1.38 |

## Per-criterion mean scores (0 to 2)

| Arm | c1 mobile→payment | c2 paid-social→cart | c3 CX region | c4 no-fabrication | c5 numerical | c6 calibrated |
|---|---:|---:|---:|---:|---:|---:|
| structured | **0.00** | 0.50 | **0.00** | 1.50 | 2.00 | 2.00 |
| naive | 2.00 | 2.00 | 2.00 | 1.17 | 1.33 | 2.00 |

For comparison, the first run (commit abed816) had structured at 7.33 and naive at 11.75, with both at 2.00 on c5.

## What happened

The fix didn't move c1. The structured version still scores 0.00 on finding the mobile payment problem, same as the first run. The fix itself worked: the diagnosis now names mobile as the segment that's underperforming, instead of describing it as desktop doing better at shipping. The eval still marks it as missing.

When I read the raw runs, the reason was a naming mismatch. The structured diagnosis says something like "mobile users have a 34pp lower conversion at the shipping step." The ground truth I wrote expects the problem to be named at the payment step. Those describe the same drop from opposite sides of the shipping to payment transition, and both are accurate, but the judge sees different step names and marks it wrong.

The pipeline defines `step_conversion` at step X as the share of users at X who made it to X+1, so a drop shows up at the step users were on. The rubric names it by the step they failed to reach. Either convention is fine. Nothing made it obvious they had to match until the eval kept scoring zero for reasons that had nothing to do with whether the answer was right. Fixing it is one small change: either have the pipeline name transitions as `from_step → to_step`, or have the rubric accept both.

The naive version's lower c5 (1.33) isn't real either. I regenerated the data for every docked run and recomputed the numbers it was marked down for. All of them were correct. For seed 1002, for example, it reported signup-week conversion of 16.5%, 16.0%, 15.1% and 14.4%, and landing to product conversion going from 74.0% to 67.2%, and those are exactly the real values. The judge marked them as fabricated because they weren't in its ground-truth file, which only covers what the structured pipeline computes. This is the same kind of problem as c1: the rubric says "verify against ground truth," and the ground truth was missing things.

When the judge could score it, the naive version found all three planted patterns (c1, c2 and c3 all at 2.00). It failed to score 4 of 10 naive runs, which isn't great, and if harder-to-grade diagnoses are the ones failing, the 10.50 could be biased.

The structured version had 8 of 10 judge failures, which is its own problem. Raising the judge's max_tokens from 4000 to 8000 last time helped less than I expected. It might be something about the new `underperforming_steps` output, or it might just be small-N noise.

## Caveats

N is very small and uneven: 2 scored runs for structured and 6 for naive. Comparing totals across the two at these sizes is directional at best.

The same model writes the diagnoses and grades them (Sonnet 5), so they likely share blind spots. More in `../rubric.md`.

## What this means

I'd keep the `Findings` → LLM structure. The fix works at the analysis layer, and the pipeline now surfaces the segment you'd act on. What needs work is the eval: the rubric and the pipeline need to agree on step names, and the numerical-accuracy check needs a ground truth that covers every number a diagnosis might report.

The broader lesson: when you build an eval for your own system, the eval and the system have to agree on what to call things and what counts as checkable, or the eval reports failures that aren't about correctness at all. Both of the low scores in this run turned out to be that.

## Raw runs

`results/raw_runs.jsonl` has one line per (arm, seed) run with the full diagnosis, the judge's scores, its reason for each criterion, token counts and duration.
