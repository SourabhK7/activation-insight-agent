"""
LLM-as-judge scoring.

The judge is Claude. It receives:
1. The diagnosis text produced by one of the arms.
2. The ground truth: the planted patterns from the synthetic data generator,
   plus the true (Python-computed) conversion rates for reference.

It returns a JSON scorecard with per-criterion scores (0/1/2), a total, and
a short justification per criterion (for auditability).

We enforce structured output by instructing the model to return only JSON
and by validating the response shape. Malformed responses raise; run_eval.py
retries once, then records the run as `judge_failed`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from anthropic import Anthropic

JUDGE_MODEL_DEFAULT = "claude-sonnet-5"

JUDGE_SYSTEM_PROMPT = """You are grading a written funnel-analysis diagnosis against a rubric. You will be given the diagnosis text and the ground truth about what patterns were planted in the underlying synthetic data. Your job is to score six criteria, each 0/1/2, per the rubric supplied in the user prompt.

Return ONLY a JSON object matching the schema in the user prompt. Do not include any commentary before or after the JSON. Do not wrap in markdown code fences.
"""


JUDGE_USER_PROMPT_TEMPLATE = """RUBRIC (score each criterion 0, 1, or 2):

1. Identifies the mobile → payment issue
   - 2: explicitly says mobile users have elevated drop-off at the payment step (direction: mobile worse than desktop)
   - 1: mentions mobile is worse overall but doesn't localize to payment
   - 0: doesn't surface it, or attributes to wrong step

2. Identifies the paid-social → add-to-cart issue
   - 2: names paid-social and localizes drop-off to add-to-cart (or the transition into it)
   - 1: mentions paid-social is worse overall but doesn't localize step
   - 0: doesn't identify paid-social, or misattributes step

3. Identifies the CX regional issue
   - 2: names CX country and localizes to shipping-info step
   - 1: mentions CX as a divergent geography without localizing step
   - 0: doesn't mention CX as concerning

4. Does not fabricate patterns not in the data
   - 2: no unsupported claims; small effects (if mentioned) are hedged as small/uncertain
   - 1: one minor over-claim
   - 0: fabricates at least one substantive finding, or invents attributes not in the data

5. Numerical accuracy (compare every rate/delta/percentage in the diagnosis to ground truth)
   - 2: all numbers within 1pp of ground truth, no fabricated numbers
   - 1: one number off by 1-3pp OR one fabricated number that is directionally correct
   - 0: two or more errors, OR any error > 3pp, OR a fabricated number that misrepresents magnitude

6. Calibrated language
   - 2: uses "consistent with", "suggests", "coincides with"; avoids "caused", "drove", "led to"; recommendations are framed as "worth testing"
   - 1: mostly calibrated, one or two overclaims
   - 0: multiple causal overclaims, or presents interpretations as conclusions

GROUND TRUTH FOR THIS RUN:

Planted patterns in the synthetic data (from synthesize.py):
- Mobile users have ~35pp lower payment-step conversion than desktop.
- Paid-social users have ~28pp lower add-to-cart conversion than average.
- CX-country users have ~22pp lower shipping-info conversion than average.
- A weak signup_week / temporal effect exists but is small and should NOT be reported as a real finding.

True computed metrics (from the actual analysis pipeline that ran this specific dataset):

```json
{ground_truth_json}
```

DIAGNOSIS TO GRADE:

```
{diagnosis_text}
```

Return a JSON object with this exact schema:

{{
  "criterion_1_mobile_payment": {{"score": 0|1|2, "reason": "one-sentence justification"}},
  "criterion_2_paid_social": {{"score": 0|1|2, "reason": "one-sentence justification"}},
  "criterion_3_cx_region": {{"score": 0|1|2, "reason": "one-sentence justification"}},
  "criterion_4_no_fabrication": {{"score": 0|1|2, "reason": "one-sentence justification"}},
  "criterion_5_numerical_accuracy": {{"score": 0|1|2, "reason": "one-sentence justification, cite the specific numbers checked"}},
  "criterion_6_calibrated_language": {{"score": 0|1|2, "reason": "one-sentence justification"}},
  "total": <sum of the six scores, 0-12>
}}
"""


class JudgeError(Exception):
    """Raised when the judge response is malformed or the API fails."""


@dataclass
class Scorecard:
    per_criterion: Dict[str, Dict[str, Any]]
    total: int
    raw_response: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_criterion": self.per_criterion,
            "total": self.total,
            "raw_response": self.raw_response,
        }


def score(
    diagnosis_text: str,
    ground_truth: Dict[str, Any],
    *,
    model: str = JUDGE_MODEL_DEFAULT,
    api_key: str | None = None,
) -> Scorecard:
    """Score a single diagnosis against ground truth."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise JudgeError("No ANTHROPIC_API_KEY.")

    client = Anthropic(api_key=key)

    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        ground_truth_json=json.dumps(ground_truth, indent=2, default=str),
        diagnosis_text=diagnosis_text,
    )

    try:
        resp = client.messages.create(
            model=model,
            # Judge output budget scales with the diagnosis it has to grade —
            # long diagnoses (e.g. 4-5k chars from the naive baseline arm)
            # require heavier reasoning to check numerical accuracy against
            # ground truth, and extended-reasoning models exhaust the ceiling
            # on internal thinking before writing JSON. Observed:
            # 1500 truncated; 4000 handled structured but not naive; 8000
            # covers both. Only tokens actually used are billed.
            max_tokens=8000,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as e:
        raise JudgeError(f"Judge API call failed: {e}") from e

    text_parts = [b.text for b in resp.content if hasattr(b, "text")]
    raw = "\n".join(text_parts).strip()

    # Occasionally the model wraps in ```json ... ``` despite the instruction.
    # Strip common wrappers before parsing.
    stripped = raw
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise JudgeError(f"Judge returned non-JSON: {e}. Raw: {raw[:400]}")

    required_keys = {
        "criterion_1_mobile_payment",
        "criterion_2_paid_social",
        "criterion_3_cx_region",
        "criterion_4_no_fabrication",
        "criterion_5_numerical_accuracy",
        "criterion_6_calibrated_language",
        "total",
    }
    missing = required_keys - set(parsed.keys())
    if missing:
        raise JudgeError(f"Judge JSON missing keys: {missing}. Raw: {raw[:400]}")

    per_criterion = {k: parsed[k] for k in required_keys - {"total"}}
    total = int(parsed["total"])

    return Scorecard(per_criterion=per_criterion, total=total, raw_response=raw)
