"""
The naive-baseline arm (Arm B).

To test whether the structured "findings JSON" approach actually beats a
simpler approach, we need a fair simpler approach to compare against.

The naive baseline:
1. Takes the same raw event data.
2. Passes Claude a pre-aggregated groupby summary (step counts, and per-segment
   step counts) — NOT the pre-computed rates or the divergent-segment analysis.
3. Asks Claude to compute conversion rates and diagnose from the counts.

This isolates the design decision under test: "does moving the arithmetic
out of the LLM's hands materially change output quality?" If Arm A (structured
findings) meaningfully beats Arm B (aggregated counts, LLM computes rates),
the README's central design claim is supported.

A note on fairness: we're deliberately NOT sending raw events (which would
be tens of thousands of rows and unfair to any LLM). We're sending a
reasonable pre-aggregation. The only thing we're asking the naive arm to do
that the structured arm doesn't is compute rates — which is exactly the
capability we're testing.
"""

from __future__ import annotations

from typing import List

import pandas as pd


NAIVE_SYSTEM_PROMPT = """You are a senior product data scientist. You will be given aggregated event counts from an e-commerce funnel, and you must produce a written diagnosis of where and why users are dropping off.

Compute conversion rates from the counts. Segment the analysis by the user attributes provided. Identify the biggest drop-off step and any cohorts whose conversion is meaningfully worse than overall.

Be calibrated. Use "consistent with", "suggests", "coincides with" — not "caused", "drove", "led to". This is observational data.

Structure your output as:
- Headline (2-3 sentences)
- Funnel overview
- Segment findings (one paragraph per segment worth mentioning)
- Interpretation (2-4 sentences, hypotheses only)
- Caveats
- Suggested next steps

Do not include a preamble. Start with the Headline section.
"""


NAIVE_USER_PROMPT_TEMPLATE = """FUNNEL EVENT COUNTS

Total users who entered the funnel: {total_users:,}

Users reaching each step (in funnel order):

{overall_counts_table}

Per-segment counts at each step (attributes: {attribute_list}):

{segment_counts_table}

Diagnose this funnel. Compute conversion rates from the counts above. Identify the biggest drop-off step and any concerning segments. Follow the structure in the system prompt.
"""


def _format_overall_counts(events: pd.DataFrame, step_order: List[str]) -> str:
    """Table of unique users who reached each step."""
    lines = ["| Step | Users reaching this step |", "|---|---:|"]
    for step in step_order:
        n = events[events["step_name"] == step]["user_id"].nunique()
        lines.append(f"| {step} | {n:,} |")
    return "\n".join(lines)


def _format_segment_counts(
    events: pd.DataFrame,
    step_order: List[str],
    attribute_columns: List[str],
) -> str:
    """
    Per-segment step counts. For each attribute, list unique users per
    (attribute_value, step) combination. Skip segments with < 500 users at
    step 1 (matching the analysis threshold — the naive arm isn't given the
    threshold, but neither arm should be graded on tiny segments).
    """
    blocks = []
    for attr in attribute_columns:
        if attr not in events.columns:
            continue
        blocks.append(f"### By {attr}\n")
        step_zero_users = events[events["step_name"] == step_order[0]].groupby(attr)["user_id"].nunique()

        header = ["| Value | " + " | ".join(step_order) + " |"]
        sep = ["|---" * (len(step_order) + 1) + "|"]
        rows = [header[0], sep[0]]

        for value in sorted(step_zero_users.index.tolist()):
            if step_zero_users[value] < 500:
                continue
            counts = []
            for step in step_order:
                n = events[
                    (events["step_name"] == step) & (events[attr] == value)
                ]["user_id"].nunique()
                counts.append(f"{n:,}")
            rows.append(f"| {value} | " + " | ".join(counts) + " |")
        blocks.append("\n".join(rows))
        blocks.append("")

    return "\n".join(blocks)


def build_naive_prompt(
    events: pd.DataFrame,
    step_order: List[str],
    attribute_columns: List[str],
) -> tuple[str, str]:
    """
    Build the (system, user) prompt pair for the naive baseline arm.
    """
    total_users = events[events["step_name"] == step_order[0]]["user_id"].nunique()
    overall_table = _format_overall_counts(events, step_order)
    segment_table = _format_segment_counts(events, step_order, attribute_columns)

    user_prompt = NAIVE_USER_PROMPT_TEMPLATE.format(
        total_users=total_users,
        overall_counts_table=overall_table,
        segment_counts_table=segment_table,
        attribute_list=", ".join(attribute_columns),
    )
    return NAIVE_SYSTEM_PROMPT, user_prompt
