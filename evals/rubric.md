# Evaluation rubric

Six criteria, scored 0 / 1 / 2. Total possible: 12. All criteria are graded by an LLM judge (Claude) against a diagnosis text and the ground-truth findings from the synthetic data generator.

The judge sees the ground truth. The scored diagnosis does not (it only sees the input its arm was given).

---

## 1. Identifies the mobile → payment issue

The synthetic data plants a mobile-payment friction pattern: mobile users convert at the payment step ~35pp lower than desktop.

- **2**: Diagnosis explicitly calls out that mobile users have elevated drop-off at the payment step, and reports the direction correctly (mobile worse than desktop).
- **1**: Mentions mobile is worse overall but does not localize the issue to the payment step.
- **0**: Does not surface a mobile / payment finding, or attributes the mobile gap to the wrong step.

## 2. Identifies the paid-social → add-to-cart issue

The data plants an ad-to-landing-page mismatch for paid-social acquisition: those users drop off at the product-page → add-to-cart transition ~28pp more than average.

- **2**: Diagnosis names paid-social as the concerning source and localizes the drop-off to the add-to-cart step (or the transition into it).
- **1**: Mentions paid-social has worse conversion but does not localize to the correct step.
- **0**: Does not identify paid-social as an issue, or misattributes its drop-off to a step it doesn't plausibly affect.

## 3. Identifies the CX regional issue

Users with `country_code = "CX"` see elevated abandonment at shipping info (~22pp).

- **2**: Diagnosis surfaces the CX geography as diverging and localizes the concern to the shipping step (or gets close).
- **1**: Mentions country-level variation and specifically CX, but doesn't localize the step.
- **0**: Does not mention CX as a concerning cohort.

## 4. Does not fabricate patterns not in the data

The synthetic data has a small weekday/signup-week effect (planted intentionally as low-magnitude noise). Real DS work heavily penalizes over-claiming, so the rubric rewards *not* elevating this to a "finding."

- **2**: The diagnosis does not claim any finding whose evidence isn't in the input. Any small effects mentioned are explicitly hedged as small or uncertain.
- **1**: One minor over-claim, or claims the weekday effect as a real pattern without hedging.
- **0**: Fabricates at least one substantive finding not supported by the input, or invents user attributes that weren't in the data.

## 5. Numerical accuracy

The single most important criterion for evaluating the "Python does math, LLM does prose" design thesis. Every conversion rate, cohort delta, or percentage that appears in the diagnosis must match the ground truth to within 1 percentage point (allowing minor rounding). This is *the* failure mode the structured design is meant to prevent.

- **2**: All numbers in the diagnosis are correct to within 1pp of ground truth. No fabricated numbers.
- **1**: One number is off by 1-3pp, or one number is fabricated but is directionally correct.
- **0**: Two or more numerical errors, or any error greater than 3pp, or a fabricated number that misrepresents magnitude.

## 6. Calibrated language

Product DS communication requires distinguishing correlation from causation. This is easy to get wrong under LLM generation.

- **2**: Uses "consistent with", "suggests", "coincides with" for interpretations. Does not use "caused", "drove", "led to". Any recommendation is framed as "worth testing" or similar, not as certain fix.
- **1**: Mostly calibrated but one or two overclaims (e.g., "the mobile UI is causing the drop-off").
- **0**: Multiple causal overclaims, or presents interpretations as conclusions.

---

## Aggregation

Per-run total is the sum across 6 criteria (0–12). Arm-level score is the mean and stdev of per-run totals over N runs. Report both.

Report per-criterion breakdowns as well — an arm can win on total but lose on criterion 5 (numerical accuracy), which is the criterion that most directly bears on the design thesis.

## Limitations of LLM-as-judge

The judge is Claude. This is not a bulletproof measurement:

- The judge shares training data with the arms it's scoring, so systematic blind spots could correlate.
- The judge is more reliable at criterion 5 (numerical, verifiable against ground truth) than at criterion 6 (calibration, which requires linguistic judgment).
- Small N (10–20 runs per arm) means confidence intervals on the mean are wide.

Mitigations:
- Ground truth for criteria 1-5 is programmatically derivable from the synthetic data generator, so the judge is checking against an oracle for those, not making autonomous calls.
- Every judge output is stored in `results/raw_runs.jsonl` for auditability. If a specific score looks wrong, it's inspectable.
- Report variance, not just means. A tight loss is a tie in this framework.
