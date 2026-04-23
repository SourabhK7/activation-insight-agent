"""Tests for cohorts.py — divergent segment detection."""

from __future__ import annotations

import pandas as pd

from activation_agent.cohorts import find_divergent_segments


STEPS = ["step_1", "step_2", "step_3"]


def _build_events(good_users: int, bad_users: int) -> pd.DataFrame:
    """
    Build a dataset with two device segments:
      - desktop: high conversion (all users reach step_3)
      - mobile: low conversion (all users drop at step_2)
    """
    rows = []
    for i in range(good_users):
        uid = f"d{i}"
        rows.append((uid, "step_1", "2026-01-01T00:00:00", "desktop"))
        rows.append((uid, "step_2", "2026-01-01T00:01:00", "desktop"))
        rows.append((uid, "step_3", "2026-01-01T00:02:00", "desktop"))
    for i in range(bad_users):
        uid = f"m{i}"
        rows.append((uid, "step_1", "2026-01-01T00:00:00", "mobile"))
        # mobile users never reach step_2 or step_3
    return pd.DataFrame(rows, columns=["user_id", "step_name", "timestamp", "device"])


def test_finds_obvious_divergence():
    events = _build_events(good_users=1000, bad_users=1000)
    segments = find_divergent_segments(
        events, STEPS, attribute_columns=["device"], min_segment_size=500
    )
    # Both desktop and mobile differ from overall end-to-end (50%) by 50pp.
    assert len(segments) == 2
    values = {s.segment_value for s in segments}
    assert values == {"desktop", "mobile"}


def test_ignores_small_segments():
    events = _build_events(good_users=1000, bad_users=10)
    segments = find_divergent_segments(
        events, STEPS, attribute_columns=["device"], min_segment_size=500
    )
    # Mobile has only 10 users → should be filtered.
    # Desktop is ~99% of users, so its rate is ~ overall rate → not "divergent".
    assert all(s.segment_value != "mobile" for s in segments)


def test_divergent_steps_identified():
    events = _build_events(good_users=1000, bad_users=1000)
    segments = find_divergent_segments(
        events, STEPS, attribute_columns=["device"], min_segment_size=500
    )
    mobile = next(s for s in segments if s.segment_value == "mobile")
    # Mobile's divergence is entirely at step_1 → step_2 transition. The
    # largest divergent step should be step_1 (or step_2, which has identical
    # delta in this degenerate case — no mobile users reach it at all).
    assert len(mobile.divergent_steps) >= 1
    assert mobile.divergent_steps[0].step_name in {"step_1", "step_2"}


def test_no_divergence_returns_empty():
    # All users same segment, same behavior → no divergent segments.
    events = _build_events(good_users=1000, bad_users=0)
    segments = find_divergent_segments(
        events, STEPS, attribute_columns=["device"], min_segment_size=500
    )
    assert segments == []
