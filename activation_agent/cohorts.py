"""
Cohort / segment divergence detection.

Finds segments whose funnel behavior diverges from the REST of the
population in ways large enough to be interesting. The goal is NOT to
flag every statistically significant difference — it's to surface the
3-5 segments a product DS would actually want to look at.

Two detection modes:

- ``threshold`` (default): a segment is flagged if
  |segment_end_to_end - rest_end_to_end| >= MIN_CONVERSION_DELTA
  and the segment has at least MIN_SEGMENT_SIZE users at the first step.

- ``statistical`` (opt-in): the practical-magnitude gate above still
  applies, AND a two-proportion z-test between the segment and the rest
  of the population must clear a Bonferroni-corrected significance bar
  (alpha / N, where N is the total count of size-passing segments
  scanned in this call).

Delta convention (important):
  Deltas are computed as SEGMENT vs. REST-OF-POPULATION, not segment
  vs. the aggregate-that-includes-the-segment. This matters when the
  divergent segment is a large share of users: segment-vs-aggregate
  understates the gap because the segment pulls the aggregate toward
  itself. Comparing to "everyone else" is the honest question a DS
  actually wants answered.

Framing for the LLM:
  Each DivergentSegment carries two step-level lists:
  - ``divergent_steps`` — every step where |seg - rest| >= 2pp,
    ranked by absolute magnitude. Both directions included.
  - ``underperforming_steps`` — the subset where the segment does
    WORSE than the rest, ranked by how much worse (most negative first).
    This is what the LLM should center its narrative on. Describing a
    minority segment's "advantage" is rarely actionable; describing the
    complementary segment's "deficit" almost always is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from typing import List, Literal, Optional

import pandas as pd

from .funnel import (
    compute_funnel,
    compute_funnel_by_segment,
)

MIN_SEGMENT_SIZE = 500
MIN_CONVERSION_DELTA = 0.04  # 4 percentage points
DEFAULT_ALPHA = 0.05

DetectionMode = Literal["threshold", "statistical"]


# ---------------------------------------------------------------------------
# Statistical helpers (no scipy dependency)
# ---------------------------------------------------------------------------


def _norm_cdf(z: float) -> float:
    """Standard normal CDF using math.erf (accurate to full float precision)."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _two_proportion_z_test(a_success: int, a_n: int, b_success: int, b_n: int) -> float:
    """Two-sided z-test for equality of two proportions (pooled variance).
    Returns a two-sided p-value. Edge cases: empty group or matching proportions
    with zero variance return 1.0 (no evidence) or 0.0 as appropriate."""
    if a_n == 0 or b_n == 0:
        return 1.0
    p_a = a_success / a_n
    p_b = b_success / b_n
    p_pool = (a_success + b_success) / (a_n + b_n)
    var = p_pool * (1.0 - p_pool) * (1.0 / a_n + 1.0 / b_n)
    if var <= 0:
        return 1.0 if p_a == p_b else 0.0
    se = sqrt(var)
    z = (p_a - p_b) / se
    return 2.0 * (1.0 - _norm_cdf(abs(z)))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StepDivergence:
    step_name: str
    segment_conversion: float
    rest_conversion: float  # was overall_conversion; now the rest-of-population rate
    delta_pp: float  # segment - rest, in absolute percentage points


@dataclass
class DivergentSegment:
    segment_column: str
    segment_value: str
    segment_n: int
    segment_end_to_end: float
    rest_end_to_end: float  # was overall_end_to_end; now the rest-of-population rate
    end_to_end_delta_pp: float  # segment - rest, in absolute pp
    divergent_steps: List[StepDivergence] = field(default_factory=list)
    underperforming_steps: List[StepDivergence] = field(default_factory=list)
    p_value: Optional[float] = None
    bonferroni_alpha: Optional[float] = None

    @property
    def divergence_direction(self) -> str:
        return "lower" if self.end_to_end_delta_pp < 0 else "higher"

    @property
    def is_statistically_significant(self) -> bool:
        return (
            self.p_value is not None
            and self.bonferroni_alpha is not None
            and self.p_value < self.bonferroni_alpha
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _step_counts(funnel_summary) -> dict[str, tuple[int, int]]:
    """Return {step_name: (users_reached, users_continued)} from a FunnelSummary."""
    return {s.step_name: (s.users_reached, s.users_continued) for s in funnel_summary.steps}


def find_divergent_segments(
    events: pd.DataFrame,
    step_order: List[str],
    attribute_columns: List[str],
    *,
    user_col: str = "user_id",
    step_col: str = "step_name",
    min_segment_size: int = MIN_SEGMENT_SIZE,
    min_conversion_delta: float = MIN_CONVERSION_DELTA,
    detection_mode: DetectionMode = "threshold",
    alpha: float = DEFAULT_ALPHA,
) -> List[DivergentSegment]:
    """Scan each attribute column for segments whose funnel behavior diverges
    from the rest of the population.

    Returns segments sorted by absolute magnitude of end-to-end divergence,
    largest first. See module docstring for the delta convention and the
    difference between divergent_steps and underperforming_steps.
    """
    overall = compute_funnel(events, step_order, user_col=user_col, step_col=step_col)
    overall_step_counts = _step_counts(overall)
    overall_success = int(round(overall.total_users * overall.end_to_end_conversion))

    # First pass: gather every size-passing candidate.
    candidates: List[tuple] = []
    for attr in attribute_columns:
        if attr not in events.columns:
            continue
        segment_funnels = compute_funnel_by_segment(
            events,
            step_order,
            segment_col=attr,
            user_col=user_col,
            step_col=step_col,
            min_segment_size=min_segment_size,
        )
        for seg_value, seg_funnel in segment_funnels.items():
            # Rest-of-population arithmetic at end-to-end level.
            seg_n = seg_funnel.total_users
            seg_success = int(round(seg_n * seg_funnel.end_to_end_conversion))
            rest_n = overall.total_users - seg_n
            if rest_n <= 0:
                # This segment IS the entire population. There is no "rest"
                # to compare against, so divergence is undefined. Skip.
                continue
            rest_success = overall_success - seg_success
            rest_e2e = rest_success / rest_n
            delta = seg_funnel.end_to_end_conversion - rest_e2e
            candidates.append((attr, seg_value, seg_funnel, delta, rest_e2e, seg_success, rest_n, rest_success))

    n_tests = max(1, len(candidates))
    bonferroni_alpha = alpha / n_tests if detection_mode == "statistical" else None

    divergent: List[DivergentSegment] = []

    for (attr, seg_value, seg_funnel, delta, rest_e2e,
         seg_success, rest_n, rest_success) in candidates:
        # Practical-magnitude gate in both modes.
        if abs(delta) < min_conversion_delta:
            continue

        p_val: Optional[float] = None
        if detection_mode == "statistical":
            p_val = _two_proportion_z_test(seg_success, seg_funnel.total_users, rest_success, rest_n)
            if p_val >= bonferroni_alpha:  # type: ignore[operator]
                continue

        # Step-level divergences: compute segment vs rest at each step.
        seg_step_counts = _step_counts(seg_funnel)
        step_divs: List[StepDivergence] = []
        for step_name in step_order[:-1]:  # skip terminal step
            if step_name not in overall_step_counts or step_name not in seg_step_counts:
                continue
            overall_reached, overall_continued = overall_step_counts[step_name]
            seg_reached, seg_continued = seg_step_counts[step_name]
            rest_reached_step = overall_reached - seg_reached
            rest_continued_step = overall_continued - seg_continued
            if rest_reached_step <= 0 or seg_reached <= 0:
                continue
            seg_rate = seg_continued / seg_reached
            rest_rate = rest_continued_step / rest_reached_step
            step_delta = seg_rate - rest_rate
            if abs(step_delta) >= 0.02:  # >= 2pp step-level difference
                step_divs.append(
                    StepDivergence(
                        step_name=step_name,
                        segment_conversion=seg_rate,
                        rest_conversion=rest_rate,
                        delta_pp=step_delta,
                    )
                )
        step_divs.sort(key=lambda sd: abs(sd.delta_pp), reverse=True)

        # The subset that matters most for framing: where THIS segment
        # is dropping harder than the rest. Sorted most-negative first.
        underperforming = sorted(
            [sd for sd in step_divs if sd.delta_pp < 0],
            key=lambda sd: sd.delta_pp,
        )

        divergent.append(
            DivergentSegment(
                segment_column=attr,
                segment_value=seg_value,
                segment_n=seg_funnel.total_users,
                segment_end_to_end=seg_funnel.end_to_end_conversion,
                rest_end_to_end=rest_e2e,
                end_to_end_delta_pp=delta,
                divergent_steps=step_divs,
                underperforming_steps=underperforming,
                p_value=p_val,
                bonferroni_alpha=bonferroni_alpha,
            )
        )

    divergent.sort(key=lambda d: abs(d.end_to_end_delta_pp), reverse=True)
    return divergent
