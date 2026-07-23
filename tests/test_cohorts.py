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


# ---------------------------------------------------------------------------
# Statistical detection mode
# ---------------------------------------------------------------------------


def _noisy_null_effect_events(n_per_segment: int, seed: int) -> pd.DataFrame:
    """
    Build events where every segment has the SAME underlying conversion rate.
    Any observed segment-vs-overall gap is pure sampling noise. Statistical
    mode should filter aggressively; threshold mode may occasionally flag.
    """
    import random
    rng = random.Random(seed)
    p_step_1_to_2 = 0.60
    p_step_2_to_3 = 0.50
    rows = []
    segments = ["a", "b", "c", "d"]
    for seg in segments:
        for i in range(n_per_segment):
            uid = f"{seg}_{i}"
            rows.append((uid, "step_1", "2026-01-01T00:00:00", seg))
            if rng.random() < p_step_1_to_2:
                rows.append((uid, "step_2", "2026-01-01T00:01:00", seg))
                if rng.random() < p_step_2_to_3:
                    rows.append((uid, "step_3", "2026-01-01T00:02:00", seg))
    return pd.DataFrame(rows, columns=["user_id", "step_name", "timestamp", "device"])


def test_statistical_mode_rejects_large_obvious_effect_as_significant():
    # Same fixture as the obvious-divergence test: mobile drops 100% of users
    # at step_1, desktop passes all through. A real 50pp effect at N=1000
    # each should clear a Bonferroni-corrected alpha.
    events = _build_events(good_users=1000, bad_users=1000)
    segments = find_divergent_segments(
        events, STEPS, attribute_columns=["device"],
        min_segment_size=500, detection_mode="statistical",
    )
    assert len(segments) >= 1
    for seg in segments:
        assert seg.p_value is not None
        assert seg.bonferroni_alpha is not None
        assert seg.is_statistically_significant, (
            f"{seg.segment_value}: p={seg.p_value}, alpha={seg.bonferroni_alpha}"
        )


def test_statistical_mode_filters_more_than_threshold_mode_on_noise():
    """
    Over multiple noisy null-effect draws, statistical mode never returns
    MORE segments than threshold mode. It can (and often will) return fewer,
    because it removes practical-magnitude flags that lack statistical
    support. Aggregated to smooth over any single unlucky seed.
    """
    threshold_total = 0
    statistical_total = 0
    for seed in range(15):
        # Small n per segment so noise can occasionally cross 4pp. At n=400
        # with true rate ~0.30, one-segment SE is ~2.3pp, so 4pp crossings
        # happen a few times across 60 (seeds × segments) opportunities.
        events = _noisy_null_effect_events(n_per_segment=400, seed=seed)
        thr = find_divergent_segments(
            events, STEPS, attribute_columns=["device"],
            min_segment_size=200, detection_mode="threshold",
        )
        stat = find_divergent_segments(
            events, STEPS, attribute_columns=["device"],
            min_segment_size=200, detection_mode="statistical",
        )
        assert len(stat) <= len(thr), (
            f"seed={seed}: statistical mode returned MORE flags than threshold "
            f"mode ({len(stat)} vs {len(thr)}); it should never be more permissive."
        )
        threshold_total += len(thr)
        statistical_total += len(stat)
    # Sanity: across 15 seeds and 4 tiny segments, some noise should trigger
    # the threshold gate at least once. If not, the noise scale is off and
    # this test isn't actually exercising the comparison.
    assert threshold_total > 0, (
        "threshold mode flagged nothing across 15 seeds — the null-effect fixture "
        "isn't noisy enough to make this comparison meaningful. Adjust n_per_segment."
    )


def test_threshold_mode_leaves_p_value_none():
    """Threshold mode must not populate p_value / bonferroni_alpha."""
    events = _build_events(good_users=1000, bad_users=1000)
    segments = find_divergent_segments(
        events, STEPS, attribute_columns=["device"],
        min_segment_size=500, detection_mode="threshold",
    )
    assert len(segments) >= 1
    for seg in segments:
        assert seg.p_value is None
        assert seg.bonferroni_alpha is None
        assert seg.is_statistically_significant is False


# ---------------------------------------------------------------------------
# Vs-rest delta and underperforming_steps
# ---------------------------------------------------------------------------


def _events_with_majority_underperformer(n_bad: int, n_good: int) -> pd.DataFrame:
    """
    Build events where 'mobile' is the MAJORITY segment and drops at step_2,
    while 'desktop' (minority) sails through. Under the old segment-vs-overall
    delta, mobile's delta would be small (because mobile pulls the aggregate
    toward itself). Under segment-vs-rest, mobile's delta should be large.
    """
    rows = []
    # Mobile: reaches step_1 and stops (0% end-to-end).
    for i in range(n_bad):
        uid = f"m{i}"
        rows.append((uid, "step_1", "2026-01-01T00:00:00", "mobile"))
    # Desktop: 100% conversion end-to-end.
    for i in range(n_good):
        uid = f"d{i}"
        rows.append((uid, "step_1", "2026-01-01T00:00:00", "desktop"))
        rows.append((uid, "step_2", "2026-01-01T00:01:00", "desktop"))
        rows.append((uid, "step_3", "2026-01-01T00:02:00", "desktop"))
    return pd.DataFrame(rows, columns=["user_id", "step_name", "timestamp", "device"])


def test_delta_is_computed_vs_rest_not_vs_overall():
    """
    With 800 mobile (majority) at 0% and 200 desktop (minority) at 100%:
    - Old delta (vs. overall 20%): mobile = -20pp, desktop = +80pp
    - New delta (vs. rest):       mobile = -100pp, desktop = +100pp
    The new comparison makes the actionable segment (mobile) at least as
    prominent as the minority segment, not less.
    """
    events = _events_with_majority_underperformer(n_bad=800, n_good=200)
    segments = find_divergent_segments(
        events, STEPS, attribute_columns=["device"], min_segment_size=100,
    )
    by_value = {s.segment_value: s for s in segments}
    assert "mobile" in by_value and "desktop" in by_value
    # Mobile is 0% end-to-end; rest (desktop) is 100%. Delta must be -1.0.
    assert abs(by_value["mobile"].end_to_end_delta_pp - (-1.0)) < 1e-6, (
        f"mobile delta = {by_value['mobile'].end_to_end_delta_pp}, expected -1.0"
    )
    # Desktop is 100%; rest (mobile) is 0%. Delta must be +1.0.
    assert abs(by_value["desktop"].end_to_end_delta_pp - 1.0) < 1e-6
    # And rest_end_to_end should reflect the opposite segment, not the aggregate.
    assert abs(by_value["mobile"].rest_end_to_end - 1.0) < 1e-6
    assert abs(by_value["desktop"].rest_end_to_end - 0.0) < 1e-6


def test_underperforming_steps_only_contains_deficits():
    """Underperforming_steps must only include steps where the segment does
    WORSE than the rest, sorted most-negative first."""
    events = _events_with_majority_underperformer(n_bad=800, n_good=200)
    segments = find_divergent_segments(
        events, STEPS, attribute_columns=["device"], min_segment_size=100,
    )
    by_value = {s.segment_value: s for s in segments}
    # Mobile drops at step_1 -> step_2. underperforming_steps must be non-empty
    # and every entry must have negative delta.
    mobile = by_value["mobile"]
    assert len(mobile.underperforming_steps) >= 1
    for sd in mobile.underperforming_steps:
        assert sd.delta_pp < 0
    # Most negative first.
    deltas = [sd.delta_pp for sd in mobile.underperforming_steps]
    assert deltas == sorted(deltas)


def test_underperforming_steps_empty_when_segment_only_outperforms():
    """A segment that BEATS the rest at every step should have no
    underperforming_steps. Desktop is that segment in this fixture."""
    events = _events_with_majority_underperformer(n_bad=800, n_good=200)
    segments = find_divergent_segments(
        events, STEPS, attribute_columns=["device"], min_segment_size=100,
    )
    by_value = {s.segment_value: s for s in segments}
    desktop = by_value["desktop"]
    assert desktop.underperforming_steps == []


def test_segment_that_is_entire_population_is_skipped():
    """When only one segment exists at all, 'rest' is empty and comparison
    is undefined. Such a segment must not appear in results."""
    # All 1000 users are 'desktop' — nobody else.
    rows = []
    for i in range(1000):
        uid = f"d{i}"
        rows.append((uid, "step_1", "2026-01-01T00:00:00", "desktop"))
        rows.append((uid, "step_2", "2026-01-01T00:01:00", "desktop"))
        rows.append((uid, "step_3", "2026-01-01T00:02:00", "desktop"))
    events = pd.DataFrame(rows, columns=["user_id", "step_name", "timestamp", "device"])
    segments = find_divergent_segments(
        events, STEPS, attribute_columns=["device"], min_segment_size=100,
    )
    assert segments == []
