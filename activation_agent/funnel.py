"""
Funnel conversion analysis.

Pure-pandas computations over the event-level funnel data. No LLM calls,
no API dependencies. The output of this module feeds into findings.py
and eventually into the LLM prompt.

Key design choice: the functions here return structured dataclasses, not
DataFrames. This is deliberate — downstream code (and the LLM prompt)
works with typed fields, not with DataFrame columns that could be
accidentally renamed or reordered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class StepMetrics:
    step_name: str
    users_reached: int
    users_continued: int  # users who reached the NEXT step
    step_conversion: float  # fraction of users_reached who continued
    drop_off: float  # 1 - step_conversion


@dataclass
class FunnelSummary:
    total_users: int
    end_to_end_conversion: float  # fraction of users who reached the last step
    steps: List[StepMetrics]
    step_order: List[str]

    @property
    def biggest_drop_off_step(self) -> StepMetrics | None:
        """Return the step with the highest drop-off, excluding the final step."""
        # The final step has no "continued" meaning, so exclude it.
        if len(self.steps) < 2:
            return None
        candidates = self.steps[:-1]
        return max(candidates, key=lambda s: s.drop_off)


def compute_funnel(
    events: pd.DataFrame,
    step_order: List[str],
    *,
    user_col: str = "user_id",
    step_col: str = "step_name",
) -> FunnelSummary:
    """
    Compute step-by-step conversion for a set of events.

    Parameters
    ----------
    events : DataFrame
        Event-level data with at least `user_col` and `step_col`.
    step_order : list of str
        Ordered list of step names defining the funnel.
    user_col, step_col : str
        Column names in `events`.

    Returns
    -------
    FunnelSummary

    Notes
    -----
    A user is considered to have "reached" a step if they have at least
    one event with that step_name. The order of their events is not checked
    — if a user has a `purchase_complete` event without a
    `shipping_info_entered` event, they still count as having reached
    `purchase_complete`. In well-formed funnel data this is a non-issue;
    in messy data it can inflate later-step counts. For this prototype we
    accept the simplification and document it.
    """
    if not step_order:
        raise ValueError("step_order must be non-empty")

    # For each step, the set of users who reached it.
    step_users = {
        step: set(events.loc[events[step_col] == step, user_col].unique())
        for step in step_order
    }

    total_users = len(step_users[step_order[0]])
    if total_users == 0:
        raise ValueError(
            f"No users reached the first step '{step_order[0]}'. "
            "Check that step_order matches the values in the data."
        )

    step_metrics: List[StepMetrics] = []
    for i, step in enumerate(step_order):
        reached = len(step_users[step])
        if i < len(step_order) - 1:
            next_step_users = step_users[step_order[i + 1]]
            continued = len(step_users[step] & next_step_users)
            conv = continued / reached if reached > 0 else 0.0
        else:
            continued = reached  # no next step; drop-off not meaningful here
            conv = 1.0
        step_metrics.append(
            StepMetrics(
                step_name=step,
                users_reached=reached,
                users_continued=continued,
                step_conversion=conv,
                drop_off=1.0 - conv,
            )
        )

    end_to_end = len(step_users[step_order[-1]]) / total_users

    return FunnelSummary(
        total_users=total_users,
        end_to_end_conversion=end_to_end,
        steps=step_metrics,
        step_order=step_order,
    )


def compute_funnel_by_segment(
    events: pd.DataFrame,
    step_order: List[str],
    segment_col: str,
    *,
    user_col: str = "user_id",
    step_col: str = "step_name",
    min_segment_size: int = 200,
) -> dict[str, FunnelSummary]:
    """
    Compute funnel metrics for each value in `segment_col`.

    Segments with fewer than `min_segment_size` users at step 1 are skipped
    to avoid noisy segment conclusions.

    The segment value is taken from the row of each user's first event for
    that step. This means if a user's attribute differs across rows (e.g.,
    they switched devices mid-funnel), the earliest-observed value wins.
    For well-formed event data this is a no-op; for messy data it makes the
    behavior deterministic.
    """
    # Determine each user's segment value from their first row.
    first_row_per_user = (
        events.sort_values("timestamp")
        .drop_duplicates(subset=[user_col], keep="first")[[user_col, segment_col]]
    )
    user_segment = dict(zip(first_row_per_user[user_col], first_row_per_user[segment_col]))

    # Tag every event with its user's segment.
    events_with_seg = events.copy()
    events_with_seg["_segment"] = events_with_seg[user_col].map(user_segment)

    result: dict[str, FunnelSummary] = {}
    for seg_value, seg_events in events_with_seg.groupby("_segment", dropna=True):
        step_one_users = seg_events.loc[
            seg_events[step_col] == step_order[0], user_col
        ].nunique()
        if step_one_users < min_segment_size:
            continue
        try:
            result[str(seg_value)] = compute_funnel(
                seg_events, step_order, user_col=user_col, step_col=step_col
            )
        except ValueError:
            continue
    return result
