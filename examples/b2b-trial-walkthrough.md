# Example: a B2B SaaS trial funnel

The main demo uses a 6-step e-commerce checkout. This one runs the agent on a different shape of funnel, a B2B SaaS free trial that ends in a paid conversion, to show it isn't hard-wired to the demo data. The steps, user attributes and planted patterns are all different, and none of the agent code changes.

## The funnel

Steps, in order:
1. `trial_started`
2. `invited_teammate`
3. `connect_data_source`
4. `run_first_workflow`
5. `hit_activation_metric`
6. `paid_conversion`

User attributes:
- `plan_tier`: starter, growth or enterprise
- `region`: NA, EMEA, LATAM or APAC
- `initial_seats_claimed`: 1 to 25
- `signup_week`: 0 to 3

## What's planted in the data (the agent doesn't see this)

A. Enterprise trials with fewer than 5 initial seats are much less likely to get through `connect_data_source`. In practice, one person evaluating an enterprise product alone often never gets far enough to see the value.

B. Trials that never invite a second teammate do worse at `run_first_workflow`, on every tier.

C. LATAM trials have slightly higher drop-off at `connect_data_source`, about 7 points. I made this one small on purpose (the demo's patterns are 20 to 35 points) to see whether the agent either misses it or makes too much of it.

## Running it

```bash
# 1. Generate the B2B dataset
python -m activation_agent generate-data \
    --preset b2b-trial \
    --n-users 20000 \
    --output data/b2b_trial_funnel.csv

# 2. Run the agent with the new step order and attributes
python -m activation_agent run \
    --data data/b2b_trial_funnel.csv \
    --steps "trial_started,invited_teammate,connect_data_source,run_first_workflow,hit_activation_metric,paid_conversion" \
    --attributes "plan_tier,region,initial_seats_claimed,signup_week" \
    --output examples/b2b-trial-diagnosis.md
```

Only the step order and the attribute list change on the command line.

## What a good diagnosis should say

- The biggest drop-off step. Here that's usually `connect_data_source` because of pattern A, but sometimes `invited_teammate` (its base rate is 0.60 with nothing planted on it). Which one wins depends on how the segments mix.
- Pattern A, clearly: enterprise trials with few seats fail at connecting a data source. It's the biggest planted effect (about 40 points), so missing it means something's broken.
- Pattern B: trials with no teammate do worse at the first workflow, named at the right step. It's the second biggest (about 25 points).
- Pattern C as a small thing worth a look, not a headline. Making a big deal of small effects is a common mistake, and the prompt's rule about calibrated language is there to prevent it.

## Why I added it

All three patterns in the main demo are 20 to 35 points, which is hard to miss. The 7-point LATAM effect here shows whether the agent stays measured when the signal isn't obvious. The `evals/` rubric scores that directly (criteria 4 and 6).

## What it doesn't show

It's still synthetic. Real warehouse data is messier: late events, steps out of order because of clock skew, missing attributes, bots inflating cohorts. On real data you'd expect to tune the prompt, the detection threshold, and the attribute list. What this example shows is narrower: the agent works on more than one funnel shape.
