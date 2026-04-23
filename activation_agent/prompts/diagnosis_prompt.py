"""
Prompt template for the activation insight diagnosis.

Design notes:

1. We pass the findings as JSON in the prompt, with explicit field descriptions
   so the LLM doesn't have to guess the schema. This is more reliable than
   passing a prose summary of the findings, because it preserves the exact
   numbers.

2. The prompt explicitly tells the LLM not to do arithmetic. All numbers
   that appear in the output must appear in the input JSON. This is the
   single most important instruction for preventing the LLM from
   hallucinating rates.

3. We ask for a specific structure (headline, then main findings, then
   caveats) rather than letting the LLM pick a format. This keeps the
   output predictable across runs.

4. The prompt forces explicit acknowledgment when the data doesn't support
   a finding — e.g., if no divergent segments were detected, the LLM must
   say that rather than inventing subtle patterns.
"""

from __future__ import annotations

import json
from typing import Dict, Any


SYSTEM_PROMPT = """You are a senior product data scientist writing up the results of a funnel analysis. Your job is to turn a structured findings object into a clear, calibrated written diagnosis that a PM or product lead could act on.

Rules — these are not negotiable:

1. ALL NUMBERS in your output must come directly from the findings JSON. Do not compute new numbers. Do not do arithmetic. If you need a number that isn't in the findings, say "the data provided doesn't include this" rather than estimating.

2. Be calibrated. You are describing correlations, not causes. Use language like "is consistent with", "suggests", or "coincides with". Do NOT use "caused", "drove", or "led to" — this is observational funnel data, not an experiment.

3. Lead with the finding, not the methodology. The PM reading this doesn't need to know how you computed conversion rates.

4. If the findings show no divergent segments, say so. Do not invent subtle patterns to make the output look richer.

5. Be specific. "Mobile users drop off" is vague. "Mobile users drop off at the payment step, 52% vs 78% desktop" is actionable.

6. End with caveats that are actually relevant — what this analysis cannot tell us. Do not include generic caveats ("data can be noisy"); include ones that matter for the decision.
"""


USER_PROMPT_TEMPLATE = """Here are the findings from a funnel analysis, serialized as JSON:

```json
{findings_json}
```

Field reference:
- `overall.total_users`: count of users who entered the funnel at step 1.
- `overall.end_to_end_conversion`: fraction of users who reached the final step (0.0 to 1.0).
- `overall.steps[i].step_conversion`: fraction of users at step `i` who continued to step `i+1`.
- `overall.steps[i].drop_off`: 1 - step_conversion. The fraction of users at step `i` who did NOT continue.
- `overall.biggest_drop_off_step`: the non-terminal step with the highest drop-off rate.
- `divergent_segments[j]`: a cohort whose end-to-end conversion differs from overall by more than the detection threshold. `segment_column` names the attribute (e.g., "device"), `segment_value` names the cohort (e.g., "mobile"), `end_to_end_delta_pp` is the absolute percentage-point difference from overall (negative = worse than overall).
- `divergent_segments[j].divergent_steps`: for this cohort, the individual steps where their conversion differs most from overall, ranked by magnitude. This tells you WHERE in the funnel the cohort's divergence is concentrated.
- `attribute_columns_analyzed`: which user attributes we examined. If an attribute isn't in this list, we didn't analyze it.
- `small_segment_counts`: per attribute, the count of segments that were too small to analyze confidently.

Write the diagnosis using this structure:

## Headline
Two to three sentences. State the overall conversion rate, the biggest drop-off step, and the single most important segment-level finding (if any). If there are no divergent segments, say so here.

## Funnel overview
A paragraph describing step-by-step conversion in prose. Highlight the biggest drop-off and any other step with drop-off above 0.40. Use the actual numbers from the findings.

## Segment findings
One short paragraph per divergent segment, in order of magnitude. For each: name the segment, its size, its end-to-end conversion vs. overall, and which step(s) explain the divergence. If there are no divergent segments, write a single sentence saying so and skip to caveats.

## What this suggests (interpretation)
2-4 sentences. Based on *where* each divergent segment diverges, offer plausible interpretations. Be explicit that these are hypotheses, not conclusions. Example: "The mobile gap being concentrated at the payment step is consistent with mobile-specific friction in the payment UI — though it could also reflect different baseline intent-to-purchase on mobile."

## Caveats
Bulleted list, 2-4 items. Include:
- That this is observational data and correlations are not causes.
- Any specific limitation from the findings (e.g., attributes that weren't analyzed, small segments).
- What would need to be tested to confirm a hypothesis.

## Suggested next steps
2-3 concrete actions a PM or DS could take based on this diagnosis. Be specific about what would be tested or investigated.

Write the diagnosis now. Do not include a preamble like "Here is the diagnosis". Just start with the Headline section.
"""


def build_prompt(findings_dict: Dict[str, Any]) -> tuple[str, str]:
    """
    Build the system and user prompts for the diagnosis API call.

    Returns (system_prompt, user_prompt).
    """
    findings_json = json.dumps(findings_dict, indent=2, default=str)
    user_prompt = USER_PROMPT_TEMPLATE.format(findings_json=findings_json)
    return SYSTEM_PROMPT, user_prompt
