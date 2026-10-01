# User Journey Analytics Agent

[![test](https://github.com/SourabhK7/activation-insight-agent/actions/workflows/test.yml/badge.svg)](https://github.com/SourabhK7/activation-insight-agent/actions/workflows/test.yml)

A Python agent that takes product data and writes up what's going on, the readout you'd otherwise spend an hour on. It has four modes: funnel drop-off, retention curves, A/B test readouts, and anomaly breakdowns.

The main design choice is that pandas does all the math and the LLM only writes. Every mode works the same way: pandas builds a structured findings object, and the LLM writes the diagnosis from that object. It never sees the raw data and is never asked to compute a number. I also tested whether that choice actually matters, and the answer surprised me (see [`evals/`](evals/) and the section below).

## How it works

Python handles conversion rates, segment breakdowns, and finding cohorts that behave differently from the overall funnel. The LLM gets a `Findings` object instead of raw events and writes a diagnosis with a fixed structure: headline, funnel overview, segments, interpretation, caveats, next steps.

I started from the common belief that LLMs are unreliable at arithmetic, so every rate the LLM computes itself is a place the pipeline could quietly mislead someone. The eval tested that, and a frontier model got the arithmetic right either way. I kept the split anyway, for a different reason: every number in the diagnosis comes from code you can inspect and test, so the output is deterministic and easy to debug. `print(findings.to_dict())` shows exactly what the LLM was given, which you can't do with a "here's the CSV, write me something" prompt.

```
funnel CSV
    │
    ▼
funnel.py + cohorts.py     ← pandas: conversion rates, segment divergence
    │
    ▼
findings.py                ← structured object (numbers live here)
    │
    ▼
diagnose.py + prompts/     ← LLM sees findings, not raw data
    │
    ▼
diagnosis.md
```

## Example output

On the included synthetic funnel:

```bash
python -m activation_agent run --data data/sample_funnel.csv --output examples/diagnosis.md
```

Part of what it wrote:

> **Headline**: Of the 50,000 users who started the checkout funnel, 26% completed purchase. The biggest drop-off is between `shipping_info_entered` and `payment_info_entered`. 38% of users at that step do not continue.
>
> **The mobile/desktop gap is the story.** Desktop users convert at 34% end to end; mobile at 19%. The gap is almost entirely at the payment step (mobile 52% completion vs. desktop 78%), which is consistent with a mobile-specific friction in the payment UI rather than a broad checkout problem.

The full output is in [examples/diagnosis.md](examples/diagnosis.md). There's also a B2B trial funnel with subtler patterns in [examples/b2b-trial-walkthrough.md](examples/b2b-trial-walkthrough.md).

## Quickstart

```bash
git clone https://github.com/SourabhK7/activation-insight-agent.git
cd activation-insight-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...  # from console.anthropic.com
```

Generate a synthetic funnel with some patterns planted in it:

```bash
python -m activation_agent generate-data \
    --n-users 50000 \
    --output data/sample_funnel.csv
```

Then run it:

```bash
python -m activation_agent run \
    --data data/sample_funnel.csv \
    --output examples/diagnosis.md
```

Each run prints the token counts, rough cost in USD, and model used, so you can see what a diagnosis costs.

The other subcommands:

- `analyze` runs only the pandas part and prints the `Findings` JSON. No LLM call and no API key needed, so it's the easiest way to see what the LLM would get.
- `generate-data --preset b2b-trial` makes a differently shaped funnel (see the B2B walkthrough).
- `retention` builds cohort retention curves and has the LLM write them up. It takes `--signups` and `--activity` CSVs, and `--cohort-column` if you want a curve per cohort. `--data-pull-date` handles cohorts that haven't been around long enough to measure yet.
- `ab-readout` analyzes a finished A/B test. Pass `--control-n --control-successes --treatment-n --treatment-successes` for a conversion metric, or a `--spec` JSON for a continuous one. Python computes the lift, 95% CI and p-value, and the LLM writes the readout.
- `anomaly` breaks down a metric that moved. Give it `--before` and `--after` CSVs (attribute columns plus `n` and `successes`) and it splits the change into segments converting differently (rate effect) versus the mix of segments changing (mix effect).

Anything that calls the LLM takes `--format full|exec|slack`. `full` is the default multi-section write-up, `exec` is three bullets under 100 words, and `slack` is a short message. It also takes `--no-llm`, which skips the API call and prints the findings as JSON.

`analyze` and `run` also take `--detection-mode threshold|statistical` (default `threshold`). The statistical mode adds a significance test on top of the size threshold, explained near the bottom.

## Using your own data

You need a CSV with three columns:

| column | type | description |
|---|---|---|
| `user_id` | string | unique identifier per user |
| `step_name` | string | which funnel step this row represents |
| `timestamp` | ISO datetime | when the step happened |

Any other columns are treated as user attributes and used for cohort breakdowns: `device`, `acquisition_source`, `country`, `signup_date`, `plan_tier`, whatever you have. More attributes give you more to look at, but past about 8 you start getting thin segments and noisy conclusions.

You also give it the step order:

```bash
python -m activation_agent run \
    --data your_funnel.csv \
    --steps "landing_page_view,product_page_view,add_to_cart,shipping_info,payment_info,purchase_complete" \
    --output diagnosis.md
```

## Does doing the math in pandas actually matter?

I could test this, so I did. [`evals/`](evals/) runs both approaches on the same synthetic data: one gives the LLM the structured findings, the other gives it aggregated counts and lets it compute the rates itself. An LLM judge scores both against a six-criterion rubric, and one of the criteria is numerical accuracy.

I ran it twice with Sonnet 5. In the first run both versions scored 2.00/2.00 on numerical accuracy. In the second, the judge gave the naive version 1.33. I went back and recomputed every number it had marked down from the same seeds, and they were all right: signup-week conversion, the CX country rates, all of it. The judge had docked numbers that weren't in its answer key, not numbers that were wrong. So "the LLM will quietly get the math wrong" didn't hold up. The naive version also scored higher overall in both runs (11.75 vs 7.33, then 10.50 vs 6.00, on the runs the judge managed to score) because its longer diagnoses picked up more of the planted patterns.

I still think the structured design is the right one. It's more predictable, it's cheaper because the prompts are shorter, and you can inspect exactly what the model was given. But the reason is determinism and debuggability, not protection from arithmetic mistakes.

The full write-up with per-criterion scores, caveats and raw data is in [`evals/results/latest.md`](evals/results/latest.md). To rerun it:

```bash
python evals/run_eval.py --n 10
```

[`evals/README.md`](evals/README.md) has the rubric, the method, and the limits of using an LLM as the judge here.

## What it doesn't do

- It doesn't do causal inference. It finds correlations, and the "why" is still up to you. The A/B mode gives you a p-value and a CI, but whether you can read it causally depends on whether the test was really randomized, whether there was a sample ratio mismatch, and so on. The agent doesn't check that.
- No attribution modeling. If a user came through several channels, it uses whatever `acquisition_source` is on the row.
- Anomaly mode won't find anomalies for you. It explains a movement you've already noticed.
- It only writes readouts. It doesn't file tickets, ping anyone, or update dashboards.
- It's not a production service. It retries with backoff on rate limits, connection errors and 5xx (via `tenacity`), and reports token counts and cost, but there's no queueing, monitoring or stored state.

## Repo layout

```
activation-insight-agent/
├── activation_agent/
│   ├── __main__.py                # CLI (all subcommands)
│   ├── funnel.py                  # step-by-step conversion math
│   ├── cohorts.py                 # segment divergence detection
│   ├── retention.py               # cohort retention curves (D1/D7/D30/…)
│   ├── ab_test.py                 # A/B two-proportion + Welch analysis
│   ├── anomaly.py                 # rate/mix decomposition
│   ├── findings.py                # structured findings dataclass (funnel)
│   ├── diagnose.py                # Anthropic API call + retry + usage tracking
│   ├── synthesize.py              # synthetic data (ecommerce, b2b-trial, retention, anomaly)
│   └── prompts/
│       ├── diagnosis_prompt.py    # funnel prompt (format-aware)
│       ├── retention_prompt.py    # retention prompt (format-aware)
│       ├── ab_readout_prompt.py   # A/B readout prompt (format-aware)
│       └── anomaly_prompt.py      # anomaly prompt (format-aware)
├── data/
│   └── sample_funnel.csv          # generated synthetic data
├── examples/
│   ├── diagnosis.md               # example output (e-commerce funnel)
│   └── b2b-trial-walkthrough.md   # different funnel shape, subtler patterns
├── evals/                         # the A/B eval of the pandas-vs-LLM math question
└── tests/
```

## A few design notes

Why a `Findings` object instead of handing the LLM the CSV? Raw event data won't fit in a prompt for any real funnel, and with a structured object you can see exactly what the LLM got and unit-test every number in it.

How segments get flagged. By default (`--detection-mode threshold`) a segment is flagged if its conversion differs from the rest by more than 4 points and it has over 500 users. That's simple and easy to explain in the write-up. `--detection-mode statistical` also runs a two-proportion z-test against the rest of the population and requires the p-value to clear a Bonferroni-corrected alpha (alpha divided by the number of segments big enough to test). It can only remove segments, never add them, so turning it on can't make things noisier. At worst the diagnosis gets shorter.

Why synthetic data? Event-level funnel data with a permissive license is hard to find, and synthetic data lets me plant patterns I know are there. That's the only way to check whether the agent finds what's really in the data rather than making up a story.

Why Sonnet as the default? It's the right cost for writing from clean structured input. This doesn't need the strongest model, Opus is overkill, and Haiku tended to hedge too much.

Sourabh Koul · [LinkedIn](https://www.linkedin.com/in/sourabhkoul/) · [GitHub](https://github.com/SourabhK7)
