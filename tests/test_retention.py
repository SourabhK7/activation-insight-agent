"""Tests for the retention analysis module."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from activation_agent.retention import compute_retention


def _make_data(n_users: int, retention_probs: dict[int, float], pull_date: date):
    """
    Build signups+activity so day-N retention is deterministically ~retention_probs[N].

    Users are evenly aged from 0..59 days old (so has_complete_window is False
    for D30 for the youngest users). Retention on day-N is planted by adding
    an event on that day for the first retention_probs[N] * n_users users
    who are old enough.
    """
    ages = [(i * 60) // n_users for i in range(n_users)]  # 0..59
    users = [f"u{i}" for i in range(n_users)]
    signups = pd.DataFrame({
        "user_id": users,
        "signup_date": [pull_date - timedelta(days=a) for a in ages],
        "cohort": ["A" if i % 2 == 0 else "B" for i in range(n_users)],
    })
    rows = []
    for day, rate in retention_probs.items():
        # Only users old enough for day-N can be planted.
        eligible = [(u, a) for u, a in zip(users, ages) if a >= day]
        n_retain = int(rate * len(eligible))
        for u, a in eligible[:n_retain]:
            sdate = pull_date - timedelta(days=a)
            rows.append({"user_id": u, "timestamp": sdate + timedelta(days=day)})
    activity = pd.DataFrame(rows)
    return signups, activity


def test_overall_curve_rates_are_computed():
    pull = date(2026, 6, 15)
    signups, activity = _make_data(6000, {1: 0.4, 7: 0.25, 30: 0.15}, pull)
    findings = compute_retention(
        signups=signups, activity=activity,
        days=[1, 7, 30],
        data_pull_date=pull,
    )
    rates = {p.day: p.rate for p in findings.overall.points}
    assert abs(rates[1] - 0.4) < 0.05
    assert abs(rates[7] - 0.25) < 0.05
    assert abs(rates[30] - 0.15) < 0.05


def test_has_complete_window_flag():
    pull = date(2026, 6, 15)
    # Users evenly aged 0..59; day-30 window is complete only for users
    # >=30 days old, which is half of them. has_complete_window should be False.
    signups, activity = _make_data(2000, {30: 0.2}, pull)
    findings = compute_retention(
        signups=signups, activity=activity,
        days=[1, 7, 30],
        data_pull_date=pull,
    )
    d30 = [p for p in findings.overall.points if p.day == 30][0]
    assert d30.has_complete_window is False
    assert d30.cohort_size_at_measurement < len(signups)

    d1 = [p for p in findings.overall.points if p.day == 1][0]
    # D1 window is complete for everyone (min age is 0, but users aged 0
    # cannot have D1 observed since pull - signup_date must be >= 1).
    # Half users would be older than 1 day.
    assert d1.cohort_size_at_measurement <= len(signups)


def test_cohort_breakdown_produces_curves():
    pull = date(2026, 6, 15)
    signups, activity = _make_data(4000, {1: 0.3, 7: 0.2}, pull)
    findings = compute_retention(
        signups=signups, activity=activity,
        days=[1, 7],
        data_pull_date=pull,
        cohort_column="cohort",
        min_cohort_size=100,
    )
    labels = {c.cohort_label for c in findings.cohorts}
    assert labels == {"A", "B"}


def test_small_cohort_is_excluded():
    pull = date(2026, 6, 15)
    signups, activity = _make_data(4000, {1: 0.3}, pull)
    findings = compute_retention(
        signups=signups, activity=activity,
        days=[1],
        data_pull_date=pull,
        cohort_column="cohort",
        min_cohort_size=10_000,  # nothing meets this
    )
    assert findings.cohorts == []
