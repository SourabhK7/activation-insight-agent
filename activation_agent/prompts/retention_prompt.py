"""
Prompt template for retention curve narratives.

Same rules as diagnosis_prompt.py: the LLM works from a structured
dict and does NOT compute new numbers.

Output format is length-adjustable via `output_format` — "full",
"exec", or "slack". The format only changes the length/register; it
does not change what numbers may be quoted (still: only ones present
in the input).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Literal

OutputFormat = Literal["full", "exec", "slack"]

SYSTEM_PROMPT = """You are a senior product data scientist writing a retention analysis for a PM or growth lead. Your job is to describe a set of retention curves in a way the reader can act on.

Rules — non-negotiable:

1. ALL NUMBERS in your output must come directly from the input JSON. Do not compute new ones.
2. Lead with the SHAPE of the curve, not the D30 endpoint. "Sharp D1 drop then flattens" is a mental picture; "18% D30" is a number without context.
3. When comparing cohorts, respect the has_complete_window flag. Do NOT compare cohorts at a day where either cohort's has_complete_window is false; if you must mention that day, explicitly say the comparison is on the older subset only.
4. Use correlational, not causal, language. "Consistent with", "suggests", "coincides with" — never "caused" or "drove".
5. Include a mechanism-to-check section: where the divergence appears in the curve points at a specific product hypothesis (early = activation/onboarding; late = engagement/habit; early-then-closes = selection). Say which one it looks like.
"""


_FORMAT_INSTRUCTIONS = {
    "full": """Write in this order (each section required unless noted):

## Shape
2-3 sentences describing how retention decays across days. Reference at least 3 of the day-N points from the overall curve. Do NOT lead with the last-day rate.

## Cohort divergence
One paragraph. If there is only one cohort (or overall-only), replace this section with a single sentence: "Only overall retention was measured; no cohort breakdown was available." Otherwise: name the cohorts that diverge, at which day the divergence appears, and by how much. If a cohort has has_complete_window=false at some day, either omit that day from the comparison or explicitly caveat it inline.

## Mechanism to check
2-4 sentences. Given where divergence appears, propose the specific product/experiment check that would explain it. Be specific about what data would settle the question.

## Caveats
Bullet list, 2-3 items. Include survivorship if any cohort has incomplete windows; observational-not-causal; and one caveat specific to what's actually in the data.
""",
    "exec": """Write a THREE-BULLET summary at exec altitude. Total length under 100 words.

- Bullet 1: shape of the curve in one sentence, with the D1 and D30 numbers from the overall curve.
- Bullet 2: the single most important cohort finding, or "no material cohort divergence" if none.
- Bullet 3: the one decision this data enables, framed as "if X, then Y".

No headings, no sections, no caveats section — but do not overclaim; if a caveat is load-bearing (e.g., partial cohort windows on the comparison), work it into the bullet inline in eight words or fewer.
""",
    "slack": """Write a Slack message. Length: 4-8 sentences. Register: casual but precise. No headings, no bullet symbols like -, use short paragraphs separated by blank lines.

Lead with the shape in one sentence, then the divergence in one sentence, then the mechanism-to-check in one or two sentences. If any cohort has incomplete windows, mention it in the same sentence you quote the affected number, in parentheses. Close with "let me know if you want me to dig into any of this" or similar (but do not use those exact words; write it naturally).
""",
}


USER_PROMPT_TEMPLATE = """Here is the retention analysis, serialized as JSON:

```json
{findings_json}
```

Field reference:
- `data_pull_date`: the ISO date used as the cutoff for observing activity. Users signed up fewer than N days before this date have has_complete_window=false at day-N.
- `days_measured`: the day-Ns computed (e.g. [1, 3, 7, 14, 30]).
- `overall`: retention across the whole signup population. `points[i]` has `day`, `retained`, `rate` (0.0-1.0), `cohort_size_at_measurement` (users old enough to observe day-N), and `has_complete_window` (True iff every signed-up user is old enough).
- `cohorts`: per-cohort curves when a cohort column was specified. Same shape as overall. Cohorts smaller than the minimum size are omitted upstream.
- `cohort_column`: which attribute defined the cohorts (null if overall-only).

{format_instructions}

Write the retention narrative now. Do not include a preamble like "Here is the analysis". Start directly with the first section.
"""


def build_prompt(findings_dict: Dict[str, Any], output_format: OutputFormat = "full") -> tuple[str, str]:
    fj = json.dumps(findings_dict, indent=2, default=str)
    fi = _FORMAT_INSTRUCTIONS.get(output_format, _FORMAT_INSTRUCTIONS["full"])
    user_prompt = USER_PROMPT_TEMPLATE.format(findings_json=fj, format_instructions=fi)
    return SYSTEM_PROMPT, user_prompt
