"""Tests for the A/B analysis module."""

from __future__ import annotations

from activation_agent.ab_test import analyze_binary, analyze_continuous


def test_binary_no_effect_is_not_significant():
    # Identical rates; p should be ~1, direction "no effect detected".
    r = analyze_binary(
        control_n=10_000, control_successes=1_000,
        treatment_n=10_000, treatment_successes=1_000,
        metric_name="conversion", hypothesis="",
    )
    assert not r.significant_at_05
    assert r.directional_interpretation == "no effect detected"
    assert abs(r.abs_lift) < 1e-9


def test_binary_large_lift_is_significant():
    # 10% vs 12% on n=10k each — comfortably significant.
    r = analyze_binary(
        control_n=10_000, control_successes=1_000,
        treatment_n=10_000, treatment_successes=1_200,
        metric_name="conversion", hypothesis="",
    )
    assert r.significant_at_05
    assert r.directional_interpretation == "treatment better"
    assert r.ci_low > 0  # CI on the lift excludes zero
    assert r.abs_lift > 0.019  # ~0.02 lift


def test_binary_treatment_worse():
    r = analyze_binary(
        control_n=5_000, control_successes=1_000,
        treatment_n=5_000, treatment_successes=800,
        metric_name="conversion", hypothesis="",
    )
    assert r.significant_at_05
    assert r.directional_interpretation == "control better"
    assert r.abs_lift < 0
    assert r.ci_high < 0  # CI entirely negative


def test_binary_small_sample_note():
    r = analyze_binary(
        control_n=50, control_successes=5,
        treatment_n=50, treatment_successes=7,
        metric_name="click", hypothesis="",
    )
    # Small-sample warnings should appear in notes.
    joined = " ".join(r.notes).lower()
    assert "fewer than 100" in joined or "fewer than 10" in joined


def test_continuous_no_effect():
    # Same mean, same sd — no effect.
    r = analyze_continuous(
        control_n=5000, control_mean=100.0, control_stddev=15.0,
        treatment_n=5000, treatment_mean=100.0, treatment_stddev=15.0,
        metric_name="arpu", hypothesis="",
    )
    assert not r.significant_at_05
    assert abs(r.abs_lift) < 1e-9


def test_continuous_positive_lift():
    r = analyze_continuous(
        control_n=5000, control_mean=100.0, control_stddev=15.0,
        treatment_n=5000, treatment_mean=101.0, treatment_stddev=15.0,
        metric_name="arpu", hypothesis="",
    )
    # Effect size 1 unit, SE ~ 0.3 → z ~ 3.3 → significant.
    assert r.significant_at_05
    assert r.directional_interpretation == "treatment better"
