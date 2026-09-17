"""
Prompt template for A/B test readouts.

Calibrated-language A/B readout, with the same math-in-Python-narrative-in-LLM
split as the rest of the repo. Output length adjusts by `output_format`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Literal

OutputFormat = Literal["full", "exec", "slack"]

SYSTEM_PROMPT = """You are a senior product data scientist writing an A/B test readout for a PM or product lead. You have been given the numbers already computed in Python; do not compute new ones.

Rules — non-negotiable:

1. Every number in your output must come from the input JSON. Do not do arithmetic.
2. Use CALIBRATED language. When p >= 0.05, say "no effect detected" or "not statistically significant" — do NOT say "no effect" or "no impact" (those are stronger claims than the data supports). When p < 0.05, say "the treatment is associated with a +X.Y pp improvement in {metric}" — do NOT say "the treatment caused" or "the treatment drove" unless the input tells you the design was clean and randomized.
3. Report the confidence interval on the lift, not just the point estimate. A wide CI that crosses zero is a critical finding.
4. Do NOT overinterpret non-significant results in either direction. A p=0.24 result is not evidence of no effect; it is evidence that this test did not have the power to see one.
5. If the input notes flag any sample-size or approximation caveats, surface them inline — do not bury them.
"""


_FORMAT_INSTRUCTIONS = {
    "full": """Write in this order:

## Bottom line
Two sentences. State the directional_interpretation, the absolute lift with its 95% CI, and the p-value.

## Result detail
One paragraph. Rate/mean in control, rate/mean in treatment, the absolute lift, the relative lift (if the input provides it), the CI, the p-value. Use the same precision the input provides.

## Interpretation
2-4 sentences. What does this mean for the hypothesis in the input? Be specific about what the data supports and what it doesn't. If the result is non-significant, do NOT conclude "no effect"; describe what the confidence interval rules in and out. If the result is significant, quote the CI to convey uncertainty in the magnitude even when the direction is clear.

## Caveats
Bullet list, 2-4 items. Include any notes from the input; whether the design supports a causal reading (from the hypothesis); and any specific limitation from the sample sizes.

## Recommended next step
One sentence. If significant with a meaningful CI, a specific action. If non-significant, either "power up and rerun" (with a rough sample-size implication) or "shelve" (with a reason).
""",
    "exec": """Write a THREE-BULLET summary at exec altitude. Under 100 words total.

- Bullet 1: directional_interpretation and the absolute lift with the CI, in one sentence.
- Bullet 2: what this means for the roadmap decision. Actionable.
- Bullet 3: the one caveat that matters most.
""",
    "slack": """Write a Slack message. 3-6 sentences. Casual but precise.

Lead with the directional interpretation and the numbers. State the CI in the same sentence or the next. If non-significant, be explicit that the test did not detect an effect (rather than "there is no effect"). Close with a one-line next step.
""",
}


USER_PROMPT_TEMPLATE = """Here is the A/B test result, serialized as JSON:

```json
{result_json}
```

Field reference:
- `metric_kind`: "binary" (rate) or "continuous" (mean).
- `metric_name`: the name of the metric that was tested.
- `hypothesis`: what the experimenter was testing.
- `control` / `treatment`: the two arms. `n` is sample size. For binary metrics, `successes` and `rate` are given. For continuous, `mean` and `stddev`.
- `abs_lift`: treatment - control in the native units of the metric (a rate difference for binary; a mean difference for continuous).
- `rel_lift`: abs_lift / control_rate (or control_mean). Null when the control value is 0.
- `ci_low` / `ci_high`: 95% CI on the ABSOLUTE lift.
- `p_value`: two-sided p-value.
- `significant_at_05`: convenience boolean.
- `directional_interpretation`: "treatment better", "control better", or "no effect detected".
- `notes`: list of caveats generated during analysis (small samples, low success counts, etc.). Surface these.

{format_instructions}

Write the readout now. Start directly; no preamble.
"""


def build_prompt(result_dict: Dict[str, Any], output_format: OutputFormat = "full") -> tuple[str, str]:
    rj = json.dumps(result_dict, indent=2, default=str)
    fi = _FORMAT_INSTRUCTIONS.get(output_format, _FORMAT_INSTRUCTIONS["full"])
    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(result_json=rj, format_instructions=fi)
