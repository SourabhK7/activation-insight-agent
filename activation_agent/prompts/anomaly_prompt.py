"""
Prompt template for anomaly-decomposition narratives.

The math (rate/mix decomposition per attribute) is already done. The
LLM writes the ranked explanation, respecting the ordering discipline
from pattern 11 in llm-ds-workflow: instrumentation checks are
NEVER concluded by the data alone, so the LLM must flag them as
external checks rather than declare them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Literal

OutputFormat = Literal["full", "exec", "slack"]

SYSTEM_PROMPT = """You are a senior product data scientist explaining a metric anomaly. You have been given a rate/mix decomposition already computed in Python; do not compute new numbers.

Rules — non-negotiable:

1. Every number in your output must appear in the input JSON.
2. Distinguish RATE effects (behavior changed within a segment) from MIX effects (segment shares of the population shifted). These have different implications: a rate effect suggests behavior/product change; a mix effect suggests marketing/acquisition/composition change.
3. Attribute the movement HONESTLY. If the largest single contribution is a mix effect, say the anomaly is largely compositional even if the writer expected a behavioral story.
4. DO NOT declare that instrumentation is or isn't broken. That's an external check the data cannot answer. Instead, in the "checks to run" section, flag instrumentation as the first thing to rule out.
5. Correlational, not causal, language. "Consistent with", "coincides with" — never "caused".
"""


_FORMAT_INSTRUCTIONS = {
    "full": """Write in this order:

## The movement
Two sentences. State the metric, the before/after aggregate rates, and the absolute delta. Use the metric_name, before_label, and after_label from the input.

## What the decomposition shows
One paragraph per attribute that has at least one meaningful contribution. Describe the top 1-2 segments by absolute total contribution (rate_effect + mix_effect), separating the rate-effect portion from the mix-effect portion. Explicitly label each as "rate" or "mix". Use the exact rate_change_pp values from the input.

## Interpretation
2-4 sentences. Given the balance of rate vs. mix effects, what kind of anomaly is this? Rate-heavy anomalies point at behavior or product changes within known segments. Mix-heavy anomalies point at acquisition or composition shifts. Both together may reflect a launch that affected some segments more than others.

## Checks to run (in order)
Numbered list of 3-5 items. Order MUST be: instrumentation checks first (data completeness, event schema, timezone alignment); then any specific hypothesis suggested by the largest rate/mix contribution; then broader behavioral hypotheses. Each check: what to look at, and what the "yes, this explains it" answer looks like.

## Caveats
Bullet list. Include: (a) that the decomposition method is a mathematical attribution and does not prove causation; (b) any attribute the LLM did NOT decompose because it wasn't in the input; (c) one specific limitation.
""",
    "exec": """Three-bullet summary. Under 100 words.

- Bullet 1: the movement in one sentence (metric, from-rate, to-rate, delta).
- Bullet 2: the single largest contributor to the change, labeled as rate or mix.
- Bullet 3: the top check to run to confirm or refute the leading hypothesis.
""",
    "slack": """Slack message, 4-7 sentences.

Lead with the movement. Second sentence: the top contributor, explicitly labeled rate vs. mix. Third-to-fifth: what this means and the top 1-2 checks to run. Casual but calibrated language throughout.
""",
}


USER_PROMPT_TEMPLATE = """Here is the anomaly decomposition, serialized as JSON:

```json
{findings_json}
```

Field reference:
- `metric_name`: the metric that moved.
- `before_label` / `after_label`: labels for the two windows being compared.
- `before_aggregate_rate` / `after_aggregate_rate`: the metric in each window (0.0-1.0 for rate metrics).
- `aggregate_delta`: after - before at the aggregate level.
- `decompositions[j].attribute`: which attribute this decomposition breaks the movement out by (e.g., "device", "acquisition_source").
- `decompositions[j].total_rate_effect`: sum of within-segment rate changes weighted by after-shares. If this dominates, the anomaly is BEHAVIORAL within known segments.
- `decompositions[j].total_mix_effect`: sum of share-of-N changes weighted by before-rates. If this dominates, the anomaly is COMPOSITIONAL.
- `decompositions[j].segments[k]`: per-segment contributions. `rate_change_pp` is the segment's rate change in percentage points; `rate_effect` and `mix_effect` are the two additive contributions. Segments are sorted by absolute total contribution (rate + mix), most impactful first.

{format_instructions}

Write the anomaly narrative now. Start directly; no preamble.
"""


def build_prompt(findings_dict: Dict[str, Any], output_format: OutputFormat = "full") -> tuple[str, str]:
    fj = json.dumps(findings_dict, indent=2, default=str)
    fi = _FORMAT_INSTRUCTIONS.get(output_format, _FORMAT_INSTRUCTIONS["full"])
    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(findings_json=fj, format_instructions=fi)
