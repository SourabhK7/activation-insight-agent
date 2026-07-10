# Worked example: B2B SaaS trial funnel

The default demo runs against a 6-step e-commerce checkout funnel. This walkthrough shows the agent applied to a different funnel shape — a B2B SaaS free-trial → paid conversion — to demonstrate it isn't just wired to the demo dataset.

The B2B funnel has different steps, different user attributes, and different planted patterns. All of which change without touching the agent code.

## Funnel shape

Steps (in order):
1. `trial_started`
2. `invited_teammate`
3. `connect_data_source`
4. `run_first_workflow`
5. `hit_activation_metric`
6. `paid_conversion`

Per-user attributes:
- `plan_tier`: starter | growth | enterprise
- `region`: NA | EMEA | LATAM | APAC
- `initial_seats_claimed`: integer (1–25)
- `signup_week`: 0–3

## Planted patterns (not shown to the agent)

**A. Solo enterprise eval collapse.** Enterprise-tier trials with fewer than 5 initial seats have a much lower `connect_data_source` completion rate. Reflects the reality that a single-person eval of an enterprise product often fails to reach the point of demonstrating value.

**B. Collaborative activation gap.** Trials that never invited a second teammate underperform at `run_first_workflow` regardless of tier.

**C. Subtle LATAM friction.** LATAM trials have modestly elevated drop-off at `connect_data_source` — ~7pp. This is intentionally subtle (harder to detect than the 20–35pp patterns in the default demo), to test whether the agent avoids either missing it *or* over-claiming it.

## Running it

```bash
# 1. Generate the B2B dataset (fresh preset added to synthesize.py).
python -m activation_agent generate-data \
    --preset b2b-trial \
    --n-users 20000 \
    --output data/b2b_trial_funnel.csv

# 2. Run the agent, telling it the new step order and the B2B attributes.
python -m activation_agent run \
    --data data/b2b_trial_funnel.csv \
    --steps "trial_started,invited_teammate,connect_data_source,run_first_workflow,hit_activation_metric,paid_conversion" \
    --attributes "plan_tier,region,initial_seats_claimed,signup_week" \
    --output examples/b2b-trial-diagnosis.md
```

The agent code doesn't change. Only the funnel shape and the attributes list change on the CLI.

## What the diagnosis should surface

A well-behaved run should:

- Correctly identify the biggest drop-off step. In this dataset it will typically be `connect_data_source` (thanks to pattern A) or `invited_teammate` (base rate is 0.60, but no planted pattern makes it worse — depending on how the segments interact, either can win the "biggest step" title).
- Explicitly call out pattern A: enterprise trials with low initial seat counts fail at data-source connection. This is the biggest planted effect (~40pp) and any agent that misses it isn't working.
- Explicitly call out pattern B: no-teammate trials underperform at first-workflow. This is the second-largest planted effect (~25pp) and should be surfaced with its correct step localization.
- **Not** confidently claim the LATAM pattern (~7pp) as a major finding. A calibrated diagnosis flags it as a minor divergence worth investigating, not as a headline result. Over-claiming small effects is a common failure mode and the calibrated-language rule in the prompt is meant to prevent it.

## Why this walkthrough matters

The default e-commerce demo has three planted patterns all in the 20–35pp range. A model would have to be very badly configured to miss patterns that large. The B2B preset intentionally includes a much subtler pattern (LATAM, 7pp) so someone reading this repo can see whether the agent stays calibrated on data that isn't a slam-dunk. The `evals/` harness scores calibration explicitly (criterion 4 and 6 in the rubric).

## What this doesn't prove

This is still synthetic data. The agent has not been demonstrated against a real B2B funnel with the compounding messiness real warehouses have (late-arriving events, users appearing in wrong step order because of clock skew, missing attribute values, bots polluting cohorts). Someone deploying this on real data should still expect to iterate on the prompt, the divergence-detection threshold, and the attribute list.

The value of this walkthrough is narrower: it proves the agent works on more than one dataset shape.
