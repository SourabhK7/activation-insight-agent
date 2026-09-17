"""Tests for the anomaly decomposition module."""

from __future__ import annotations

import pandas as pd

from activation_agent.anomaly import analyze_anomaly


def test_pure_mix_effect_is_labeled_mix():
    # Per-segment rates are identical between before and after; only the
    # share of each segment changes. Total rate_effect must be ~0.
    before = pd.DataFrame([
        {"src": "A", "n": 10_000, "successes": 500},   # rate 5%
        {"src": "B", "n": 10_000, "successes": 200},   # rate 2%
    ])
    after = pd.DataFrame([
        {"src": "A", "n":  4_000, "successes": 200},   # rate 5%
        {"src": "B", "n": 16_000, "successes": 320},   # rate 2%
    ])
    result = analyze_anomaly(
        before=before, after=after,
        attributes=["src"],
        metric_name="signup_rate",
    )
    assert len(result.decompositions) == 1
    dec = result.decompositions[0]
    # aggregate must have moved
    assert result.aggregate_delta < 0
    # rate effect near zero, mix effect explains everything
    assert abs(dec.total_rate_effect) < 1e-9
    assert abs(dec.total_mix_effect - result.aggregate_delta) < 1e-9


def test_pure_rate_effect_is_labeled_rate():
    # Shares stable, rates change.
    before = pd.DataFrame([
        {"src": "A", "n": 10_000, "successes": 500},   # 5%
        {"src": "B", "n": 10_000, "successes": 300},   # 3%
    ])
    after = pd.DataFrame([
        {"src": "A", "n": 10_000, "successes": 400},   # 4%
        {"src": "B", "n": 10_000, "successes": 300},   # 3%
    ])
    result = analyze_anomaly(
        before=before, after=after,
        attributes=["src"],
        metric_name="signup_rate",
    )
    dec = result.decompositions[0]
    assert abs(dec.total_mix_effect) < 1e-9
    # A's rate dropped by 1pp → contributes -0.005 to aggregate at ~50% share.
    assert dec.total_rate_effect < 0
    # Segments sorted by absolute total contribution: A must be first.
    assert dec.segments[0].segment_value == "A"


def test_missing_attribute_column_is_skipped():
    before = pd.DataFrame([{"src": "A", "n": 100, "successes": 10}])
    after = pd.DataFrame([{"src": "A", "n": 100, "successes": 10}])
    result = analyze_anomaly(
        before=before, after=after,
        attributes=["src", "nonexistent"],
        metric_name="rate",
    )
    assert len(result.decompositions) == 1
    assert result.decompositions[0].attribute == "src"
