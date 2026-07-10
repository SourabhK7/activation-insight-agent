# User Journey Analytics Agent

[![test](https://github.com/SourabhK7/activation-insight-agent/actions/workflows/test.yml/badge.svg)](https://github.com/SourabhK7/activation-insight-agent/actions/workflows/test.yml)

A Python analytics agent that turns raw funnel event data into a written diagnosis of where and why users are dropping off. Give it a messy CSV; get back the readout you'd otherwise spend an hour writing yourself.

The architecture is the interesting part: **pandas owns the arithmetic, the LLM owns the narrative**, and the two never mix. That's the design decision the rest of the repo is built around, and — importantly — the one the [`evals/`](evals/) directory exists to measure rather than assert.

---

## Architecture

Two layers, deliberately separated:

- **Python does the math.** Conversion rates, segment breakdowns, and the search for cohorts that diverge from the overall funnel — all pandas. The LLM never sees a number it has to compute.
- **The LLM does the writing.** It receives a structured `Findings` object, not raw events, and produces a diagnosis with a fixed structure: headline, funnel overview, segments, interpretation, caveats, next steps.

The split isn't arbitrary. LLMs are unreliable at arithmetic and reliable at prose; every conversion rate the LLM computes itself is a place the pipeline can silently lie to a stakeholder. Moving arithmetic to pandas removes that entire class of failure.

The `Findings` object is inspectable: `print(findings.to_dict())` shows exactly what the LLM was given. That makes this pipeline debuggable in ways an end-to-end "here's the CSV, write me something" prompt is not.

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

---

## Example output

Running against the included synthetic funnel:

```bash
python -m activation_agent run --data data/sample_funnel.csv --output examples/diagnosis.md
```

A snippet from the output:

> **Headline**: Of the 50,000 users who started the checkout funnel, 26% completed purchase. The biggest drop-off is between `shipping_info_entered` and `payment_info_entered` — 38% of users at that step do not continue.
>
> **The mobile–desktop gap is the story.** Desktop users convert at 34% end-to-end; mobile at 19%. The gap is almost entirely at the payment step (mobile 52% completion vs. desktop 78%), which is consistent with a mobile-specific friction in the payment UI rather than a broad checkout problem.

Full example: [examples/diagnosis.md](examples/diagnosis.md). A different-shaped B2B trial funnel walkthrough with subtler patterns: [examples/b2b-trial-walkthrough.md](examples/b2b-trial-walkthrough.md).

---

## Quickstart

```bash
git clone https://github.com/SourabhK7/activation-insight-agent.git
cd activation-insight-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...  # from console.anthropic.com
```

Generate a synthetic funnel with known planted patterns:

```bash
python -m activation_agent generate-data \
    --n-users 50000 \
    --output data/sample_funnel.csv
```

Run the full pipeline:

```bash
python -m activation_agent run \
    --data data/sample_funnel.csv \
    --output examples/diagnosis.md
```

The CLI prints token counts, an approximate USD cost, and the model used to stderr on every run, so you can see what each diagnosis costs without setting up separate instrumentation.

Two other subcommands:

- `analyze` — runs only the pandas layer and prints the `Findings` JSON. No LLM call, no API key needed. Good for inspecting what the LLM would receive.
- `generate-data --preset b2b-trial` — a differently-shaped funnel (see the B2B walkthrough).

---

## Using it on your own data

The agent expects a CSV with these three columns:

| column | type | description |
|---|---|---|
| `user_id` | string | unique identifier per user |
| `step_name` | string | which funnel step this row represents |
| `timestamp` | ISO datetime | when the step happened |

Any additional columns are treated as user attributes and used for cohort breakdowns — `device`, `acquisition_source`, `country`, `signup_date`, `plan_tier`, whatever you have. More attributes = richer cohort analysis, but beyond ~8 dimensions you start getting thin segments and noisy conclusions.

You also tell the agent the order of funnel steps:

```bash
python -m activation_agent run \
    --data your_funnel.csv \
    --steps "landing_page_view,product_page_view,add_to_cart,shipping_info,payment_info,purchase_complete" \
    --output diagnosis.md
```

---

## Does the design assumption hold?

The claim that pre-computing rates in pandas is better than letting the LLM do the arithmetic is testable. The [`evals/`](evals/) directory has an A/B evaluation — same synthetic data, two prompt strategies (structured findings vs. aggregated counts with LLM computing rates), scored by an LLM judge against a 6-criterion rubric that includes numerical accuracy.

```bash
python evals/run_eval.py --n 10
```

Results land in `evals/results/latest.md`. If the structured pipeline doesn't win on numerical accuracy specifically, the design should be simplified. See [`evals/README.md`](evals/README.md) for how to run and interpret it.

---

## Scope

Being explicit about what this isn't:

- **No causal inference.** The agent surfaces correlations. It does not tell you *why*. That's your job.
- **No attribution modeling.** If a user has multiple acquisition sources, the row's `acquisition_source` is used as-is.
- **No week-over-week comparison built in.** The agent treats the data as a single snapshot. Compare weeks by running it twice.
- **No automatic action.** Produces a readout; does not file tickets, ping PMs, or update dashboards.
- **Not a production service.** There's basic retry with exponential backoff on rate limits, connection errors, and 5xx (via `tenacity`), and every run reports token counts and an estimated cost. But there's no queueing, no observability, no persistent state. It's a tool for DS workflows, not a service.

---

## Repo layout

```
activation-insight-agent/
├── activation_agent/
│   ├── __main__.py                # CLI
│   ├── funnel.py                  # step-by-step conversion math
│   ├── cohorts.py                 # segment divergence detection
│   ├── findings.py                # structured findings dataclass
│   ├── diagnose.py                # Anthropic API call + retry + usage tracking
│   ├── synthesize.py              # synthetic data generators (ecommerce, b2b-trial)
│   └── prompts/
│       └── diagnosis_prompt.py    # the prompt template
├── data/
│   └── sample_funnel.csv          # generated synthetic data
├── examples/
│   ├── diagnosis.md               # example output (e-commerce funnel)
│   └── b2b-trial-walkthrough.md   # different funnel shape, subtler patterns
├── evals/                         # A/B evaluation harness for the design decision
└── tests/
```

---

## Notes on some of the choices

**Why a structured `Findings` object instead of passing raw CSV to the LLM?** Because the LLM would then have to compute aggregates, which it does unreliably. A structured intermediate also keeps the prompt small enough to stay well under context limits on huge funnels.

**Why detect divergent segments with a simple threshold rather than a proper statistical test?** The prototype uses `|segment_rate − overall_rate| > 4pp AND segment_n > 500`. A production tool should use a proportion z-test with Bonferroni correction. This is intentionally simple because (a) the agent's output is a starting point for human review, not a final decision, and (b) adding statistical rigor is a real project, not an afternoon. See `cohorts.py` for the precise rule.

**Why synthetic data?** Public event-level funnel data with permissive licensing is hard to come by, and synthesizing lets you plant known patterns. Planting known patterns means you can check whether the agent surfaces what's actually there instead of inventing stories from noise. Real data would not allow this check.

**Why Claude Sonnet as the default?** Quality-cost fit for narrative generation from structured input. The task doesn't need frontier reasoning; it needs reliable prose from a clean schema. Opus is overkill; Haiku sometimes over-hedges.

---

## Author

Sourabh Koul — Data Scientist, San Jose CA. [LinkedIn](https://www.linkedin.com/in/sourabhkoul/) · [GitHub](https://github.com/SourabhK7)
