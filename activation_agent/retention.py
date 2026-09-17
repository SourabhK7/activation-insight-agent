"""
Cohort retention analysis.

Same architectural principle as funnel.py: pandas owns the arithmetic,
this module returns structured dataclasses (never DataFrames) that flow
into the Findings object and then to the LLM.

Retention here is defined as: "the user had at least one event of any
kind on day D relative to their signup day, where D is measured in
whole days." That's the operational definition; alternate definitions
(specific-event retention, session-based, etc.) can be plugged in by
filtering the events DataFrame before calling this module.

The one non-obvious design choice: the cohort_size at each day-N is
the number of users whose signup date is at least N days before the
data_pull_date. That means the D30 rate for a cohort partly younger
than 30 days is computed on the *older subset* of the cohort — a form
of survivorship. We surface this explicitly in the returned dataclass
(via `has_complete_window`) so the LLM can quote or hedge the numbers
correctly. See `evals/` for how we test that hedging actually happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

DEFAULT_RETENTION_DAYS = [1, 3, 7, 14, 30]


@dataclass
class RetentionPoint:
    """Retention rate at a specific day-N for a cohort."""
    day: int
    cohort_size_at_measurement: int  # users old enough to have this day-N observed
    retained: int
    rate: float
    has_complete_window: bool  # True iff ALL users in the cohort are >=N days old


@dataclass
class CohortRetention:
    """Retention curve for one cohort."""
    cohort_label: str
    cohort_size: int  # signups in this cohort
    points: List[RetentionPoint] = field(default_factory=list)


@dataclass
class RetentionFindings:
    """Bundle returned to Findings/LLM."""
    data_pull_date: str  # ISO date used for age calculations
    days_measured: List[int]
    overall: CohortRetention
    cohorts: List[CohortRetention] = field(default_factory=list)
    cohort_column: Optional[str] = None  # None when overall-only


def _to_date(x) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    return pd.to_datetime(x).date()


def _compute_curve(
    signups: pd.DataFrame,
    activity: pd.DataFrame,
    days: List[int],
    data_pull_date: date,
    *,
    user_col: str = "user_id",
    signup_col: str = "signup_date",
    event_time_col: str = "timestamp",
) -> List[RetentionPoint]:
    """
    Compute retention at each day-N for a signup set.

    Parameters
    ----------
    signups : DataFrame
        One row per user, with `user_col` and `signup_col`.
    activity : DataFrame
        Event rows with `user_col` and `event_time_col`. Any event counts
        as retention on the day it occurred.
    days : list of int
        Day-Ns to measure.
    data_pull_date : date
        The cutoff for observing activity. Users signed up fewer than N
        days before this date do not have a complete day-N window.
    """
    if signups.empty:
        return [
            RetentionPoint(day=d, cohort_size_at_measurement=0, retained=0, rate=0.0, has_complete_window=False)
            for d in days
        ]

    signup_map = dict(zip(signups[user_col], pd.to_datetime(signups[signup_col]).dt.date))

    # Precompute activity dates per user as a set for fast membership.
    activity = activity.copy()
    activity["_evdate"] = pd.to_datetime(activity[event_time_col]).dt.date
    activity_by_user: Dict[str, set] = (
        activity.groupby(user_col)["_evdate"].apply(set).to_dict()
    )

    out: List[RetentionPoint] = []
    for d in days:
        # Users with enough elapsed time to observe day-d.
        eligible_users = [
            u for u, sdate in signup_map.items()
            if (data_pull_date - sdate).days >= d
        ]
        n_eligible = len(eligible_users)
        retained = 0
        for u in eligible_users:
            sdate = signup_map[u]
            target = sdate + timedelta(days=d)
            evs = activity_by_user.get(u, set())
            if target in evs:
                retained += 1
        rate = retained / n_eligible if n_eligible > 0 else 0.0
        has_complete = n_eligible == len(signup_map)
        out.append(
            RetentionPoint(
                day=d,
                cohort_size_at_measurement=n_eligible,
                retained=retained,
                rate=rate,
                has_complete_window=has_complete,
            )
        )
    return out


def compute_retention(
    signups: pd.DataFrame,
    activity: pd.DataFrame,
    *,
    days: Optional[List[int]] = None,
    data_pull_date=None,
    cohort_column: Optional[str] = None,
    min_cohort_size: int = 500,
    user_col: str = "user_id",
    signup_col: str = "signup_date",
    event_time_col: str = "timestamp",
) -> RetentionFindings:
    """
    Compute the overall retention curve and (optionally) a per-cohort
    breakdown, returning a RetentionFindings object.
    """
    days = days or DEFAULT_RETENTION_DAYS
    if data_pull_date is None:
        # Default to the max event date in the data.
        data_pull_date = pd.to_datetime(activity[event_time_col]).dt.date.max()
    else:
        data_pull_date = _to_date(data_pull_date)

    overall_points = _compute_curve(
        signups, activity, days, data_pull_date,
        user_col=user_col, signup_col=signup_col, event_time_col=event_time_col,
    )
    overall = CohortRetention(
        cohort_label="overall",
        cohort_size=len(signups),
        points=overall_points,
    )

    cohort_curves: List[CohortRetention] = []
    if cohort_column and cohort_column in signups.columns:
        for label, seg_signups in signups.groupby(cohort_column, dropna=True):
            if len(seg_signups) < min_cohort_size:
                continue
            seg_users = set(seg_signups[user_col])
            seg_activity = activity[activity[user_col].isin(seg_users)]
            pts = _compute_curve(
                seg_signups, seg_activity, days, data_pull_date,
                user_col=user_col, signup_col=signup_col, event_time_col=event_time_col,
            )
            cohort_curves.append(
                CohortRetention(
                    cohort_label=str(label),
                    cohort_size=len(seg_signups),
                    points=pts,
                )
            )

    return RetentionFindings(
        data_pull_date=str(data_pull_date),
        days_measured=days,
        overall=overall,
        cohorts=cohort_curves,
        cohort_column=cohort_column,
    )
