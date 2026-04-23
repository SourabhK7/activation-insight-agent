# User Journey Analytics Agent

A Python agent that ingests e-commerce funnel data, runs cohort analysis, and produces a written diagnosis of where and why users are dropping off.

The goal is practical: given a messy funnel CSV, produce a readout a product DS would write for a PM in 5 minutes — without the DS having to write it.

---

## What it actually does

1. **Ingests** a funnel CSV (one row per event, with user_id, step_name, timestamp, and arbitrary user attributes for cohorting).
2. **Computes** step-over-step conversion rates, drop-off magnitudes, and cohort breakdowns across supplied user attributes (device, acquisition source, country, etc.).
3. **Detects** interesting patterns: steps with unusually high drop-off, segments where conversion diverges from overall, and temporal shifts.
4. **Writes** a structured diagnosis using the Anthropic API. The LLM receives the structured analysis output (not the raw data) and produces the narrative.

Division of responsibilities:

- **Python does the math.** Conversion rates, segment breakdowns, statistical comparisons are all in pandas. The LLM never sees raw numbers it has to compute.
- **The LLM does the narrative.** Interpretation, prioritization of findings, and prose are LLM-generated from a structured input.

This split matters because LLMs are unreliable at arithmetic and very reliable at prose. Letting the LLM compute a conversion rate is where agents go wrong.

---

## Example output

Running against the included synthetic e-commerce funnel:

```bash
$ python -m activation_agent run --data data/sample_funnel.csv --output examples/diagnosis.md
```

Produces a diagnosis like:

> **Headline**: Of the 50,000 users who started the checkout funnel, 26% completed purchase. The single biggest drop-off is between "shipping_info_entered" and "payment_info_entered" (38% of users abandoning at this step), substantially higher than industry norms and higher than any other single step.
>
> **The mobile-desktop gap is the story.** Desktop users convert at 34% end-to-end; mobile users at 19%. The gap is almost entirely at the payment step: mobile payment completion is 52% vs. desktop 78%. This is consistent with a mobile-specific friction in the payment UI rather than a broad checkout problem.
>
> **Paid-social acquired users are a second concern.** Users acquired through paid social convert at 14%, vs. 29% for organic and 31% for paid search. Their drop-off is earlier in the funnel (product page view to add-to-cart), which is consistent with a mismatch between ad creative and the landing experience rather than a checkout problem.
>
> [...continues with specific recommendations and caveats...]

The full example output is at [examples/diagnosis.md](examples/diagnosis.md).

---

## Quickstart

### Setup

```bash
git clone https://github.com/SourabhK7/activation-insight-agent.git
cd activation-insight-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...  # get a key at console.anthropic.com
```

### Generate synthetic data

```bash
python -m activation_agent generate-data \
    --n-users 50000 \
    --output data/sample_funnel.csv
```

This produces a realistic e-commerce funnel with intentionally-planted drop-off patterns (a mobile payment issue, a paid-social targeting issue, and a regional checkout slowdown). Planting known patterns lets you verify the agent actually surfaces them.

### Run the diagnosis

```bash
python -m activation_agent run \
    --data data/sample_funnel.csv \
    --output examples/diagnosis.md
```

The diagnosis is written to stdout and (if `--output` is passed) to the file.

---

## Use on your own data

The agent expects a CSV with at least these columns:

| column | type | description |
|---|---|---|
| `user_id` | string | unique identifier per user |
| `step_name` | string | which funnel step this row represents |
| `timestamp` | ISO datetime | when the step happened |

Any additional columns are treated as user attributes and used for cohort breakdowns. Useful attributes include `device`, `acquisition_source`, `country`, `signup_date`, `plan_tier`. The more attributes you include, the richer the cohort analysis — but too many dimensions (> 8) leads to thin segments and noisy conclusions.

You also need to tell the agent the order of funnel steps. Define it in a YAML config or pass inline:

```bash
python -m activation_agent run \
    --data your_funnel.csv \
    --steps "landing_page_view,product_page_view,add_to_cart,shipping_info,payment_info,purchase_complete" \
    --output diagnosis.md
```

---

## Architecture

```
┌──────────────────┐
│  Funnel CSV      │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│  funnel.py               │  Conversion rates per step, per segment.
│  (pandas)                │  Pure Python math, no LLM.
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  cohorts.py              │  Finds segments with divergent conversion.
│  (pandas + scipy)        │  Uses a simple statistical screen: segments
│                          │  with at least N users AND at least X pp
│                          │  delta from overall.
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  findings.py             │  Structured findings object:
│                          │  - overall funnel metrics
│                          │  - ranked drop-off steps
│                          │  - ranked divergent segments
│                          │  - temporal patterns (if data spans >7 days)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  diagnose.py             │  Sends findings (not raw data) to Claude
│  (Anthropic API)         │  via a structured prompt. Returns the
│                          │  written diagnosis.
└────────┬─────────────────┘
         │
         ▼
┌──────────────────┐
│  Diagnosis .md   │
└──────────────────┘
```

### Why this separation

The most common agent failure mode is letting the LLM do work it's unreliable at. Arithmetic, aggregation, and statistical comparison are *unreliable* LLM skills. Prose generation, prioritization, and calibrated interpretation are *reliable* LLM skills.

By doing all the math in pandas and sending only the structured findings to Claude, we avoid the whole class of "the LLM confidently computed the wrong conversion rate" bugs. Claude's job is to decide which findings matter and to write about them well.

A secondary benefit: the findings object is inspectable. You can `print(findings.to_dict())` to see exactly what the LLM was given. This makes debugging much easier than an end-to-end LLM pipeline.

---

## What this agent does NOT do

Being honest about scope:

- **No causal inference.** The agent surfaces correlations ("mobile users drop off at payment"). It does not tell you *why*. That's your job after reading the diagnosis.
- **No attribution modeling.** If the same user appears in multiple acquisition sources, the agent uses whatever source is in the row. Attribution is out of scope.
- **No temporal experiments.** The agent treats the data as a single snapshot. If you want to compare week-over-week, run it twice and diff the diagnoses.
- **No automatic action.** The agent produces a readout; it does not file tickets, ping PMs, or update dashboards. That's a different project.
- **Not production-hardened.** No retries on API failures, no rate-limit handling beyond what the SDK provides, no observability. This is a tool for DS workflows, not a service.

---

## Structure

```
activation-insight-agent/
├── README.md                      # this file
├── requirements.txt
├── LICENSE
├── activation_agent/
│   ├── __init__.py
│   ├── __main__.py                # CLI entry point
│   ├── funnel.py                  # step-by-step conversion math
│   ├── cohorts.py                 # segment divergence detection
│   ├── findings.py                # structured findings dataclass
│   ├── diagnose.py                # Anthropic API call + prompt
│   ├── synthesize.py              # synthetic data generator
│   └── prompts/
│       └── diagnosis_prompt.py    # the prompt template
├── data/
│   └── sample_funnel.csv          # generated synthetic data
├── examples/
│   └── diagnosis.md               # example output
└── tests/
    ├── test_funnel.py
    └── test_cohorts.py
```

---

## Design decisions worth noting

**Why a structured `Findings` object instead of passing raw CSV to the LLM?**
Because the LLM would then have to compute aggregates, which it does unreliably. A structured intermediate also means the prompt is small enough to stay well under context limits even on huge funnels.

**Why detect divergent segments with a simple threshold rather than a proper statistical test?**
The prototype uses `|segment_rate - overall_rate| > 4pp AND segment_n > 500` as a screen. This is a deliberate simplification — a real product DS tool should use a proportion z-test with Bonferroni correction. I kept it simple because (a) the agent's output is a starting point for human review, not a final decision, and (b) adding statistical rigor here is a real project, not an afternoon addition. See `cohorts.py` docstring for the precise rule.

**Why synthetic data and not a real public dataset?**
Public e-commerce funnels either don't exist at the granularity needed (event-level) or come with terms of use that make them awkward to ship in a demo repo. Synthesizing the data lets me *plant known truth* — I can verify the agent actually surfaces the patterns I intentionally baked in. Real data would not allow this validation.

**Why Claude Sonnet as the default model?**
Sonnet is the right quality/cost tradeoff for narrative generation from structured input. The task does not need frontier-level reasoning; it needs reliable prose from a clean schema. Opus is overkill; Haiku occasionally over-hedges on the narrative.

---

## Author

Sourabh Koul — Data Scientist, San Jose CA
[LinkedIn](https://www.linkedin.com/in/sourabhkoul/) · [GitHub](https://github.com/SourabhK7)
