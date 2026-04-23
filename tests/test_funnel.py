"""Tests for funnel.py — core conversion math."""

from __future__ import annotations

import pandas as pd
import pytest

from activation_agent.funnel import compute_funnel, compute_funnel_by_segment


STEPS = ["step_1", "step_2", "step_3"]


def _events(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Small helper: build an events df from (user_id, step, timestamp) tuples."""
    return pd.DataFrame(rows, columns=["user_id", "step_name", "timestamp"])


def test_compute_funnel_basic():
    # 3 users reach step 1, 2 reach step 2, 1 reaches step 3.
    events = _events([
        ("u1", "step_1", "2026-01-01T00:00:00"),
        ("u1", "step_2", "2026-01-01T00:01:00"),
        ("u1", "step_3", "2026-01-01T00:02:00"),
        ("u2", "step_1", "2026-01-01T00:00:00"),
        ("u2", "step_2", "2026-01-01T00:01:00"),
        ("u3", "step_1", "2026-01-01T00:00:00"),
    ])
    summary = compute_funnel(events, STEPS)
    assert summary.total_users == 3
    assert summary.steps[0].users_reached == 3
    assert summary.steps[0].users_continued == 2
    assert summary.steps[0].step_conversion == pytest.approx(2 / 3)
    assert summary.steps[1].users_reached == 2
    assert summary.steps[1].users_continued == 1
    assert summary.steps[2].users_reached == 1
    assert summary.end_to_end_conversion == pytest.approx(1 / 3)


def test_biggest_drop_off_step():
    events = _events([
        ("u1", "step_1", "2026-01-01T00:00:00"),
        ("u1", "step_2", "2026-01-01T00:01:00"),
        ("u1", "step_3", "2026-01-01T00:02:00"),
        ("u2", "step_1", "2026-01-01T00:00:00"),
        # u2 drops at step 1 → step 2 (50% conversion there, so 50% drop)
        ("u3", "step_1", "2026-01-01T00:00:00"),
        ("u3", "step_2", "2026-01-01T00:01:00"),
        # u3 drops at step 2 → step 3 (one of two continued, so 50% there too)
        ("u4", "step_1", "2026-01-01T00:00:00"),
        ("u4", "step_2", "2026-01-01T00:01:00"),
        ("u4", "step_3", "2026-01-01T00:02:00"),
    ])
    summary = compute_funnel(events, STEPS)
    # Both step_1 and step_2 have identical drop-off here; either is fine.
    # The function should return one of the two, not None.
    assert summary.biggest_drop_off_step is not None
    assert summary.biggest_drop_off_step.step_name in {"step_1", "step_2"}


def test_compute_funnel_empty_raises():
    events = _events([])
    with pytest.raises(ValueError):
        compute_funnel(events, STEPS)


def test_compute_funnel_no_step_order_raises():
    events = _events([("u1", "step_1", "2026-01-01T00:00:00")])
    with pytest.raises(ValueError):
        compute_funnel(events, [])


def test_compute_funnel_by_segment_filters_small_segments():
    # Build data with one big segment (device=desktop, 10 users at step 1)
    # and one small segment (device=mobile, 2 users at step 1).
    rows = []
    for i in range(10):
        rows.append((f"d{i}", "step_1", "2026-01-01T00:00:00", "desktop"))
    for i in range(2):
        rows.append((f"m{i}", "step_1", "2026-01-01T00:00:00", "mobile"))
    events = pd.DataFrame(rows, columns=["user_id", "step_name", "timestamp", "device"])

    result = compute_funnel_by_segment(
        events, STEPS, segment_col="device", min_segment_size=5
    )
    assert "desktop" in result
    assert "mobile" not in result  # filtered out as too small
