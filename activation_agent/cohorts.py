"""
Cohort / segment divergence detection.

Finds segments whose funnel behavior diverges from the overall population
in ways large enough to be interesting. The goal is NOT to flag every
statistically significant difference — it's to surface the 3-5 segments
a product DS would actually want to look at.

Two detection modes:

- ``threshold`` (default): a segment is flagged if
  |segment_end_to_end - overall_end_to_end| >= MIN_CONVERSION_DELTA
  and the segment has at least MIN_SEGMENT_SIZE users at the first step.
  Simple, easy to explain in the diagnosis, hard to misuse.

- ``statistical`` (opt-in): the practical-magnitude gate above still
  applies, AND a two-proportion z-test between the segment and the rest
  of the population must clear a Bonferroni-corrected significance bar
  (alpha / N, where N is the total count of size-passing segments
  scanned in this call). This is what lets you distinguish "500 users,
  4pp gap, could be noise" from "50,000 users, 4pp gap, essentially
  certain to be real."

Statistical mode never flags MORE than threshold mode — it can only
remove candidates that fail the significance bar. That means turning
it on is safe: worst case, the diagnosis gets shorter.

Per divergent segment, we also identify which step(s) explain the
divergence by comparing step-by-step conversion. Step-level divergences
are descriptive (not a gate) in both modes.
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
    """
    Two-sided z-test for equality of two proportions (pooled variance).
    Returns a two-sided p-value.

    Edge cases:
      - Either group empty → p = 1.0 (no evidence).
      - Both proportions equal AND standard error is zero → p = 1.0.
    """
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
    overall_conversion: float
    delta_pp: float  # segment - overall, in absolute percentage points


@dataclass
class DivergentSegment:
    segment_column: str
    segment_value: str
    segment_n: int
    segment_end_to_end: float
    overall_end_to_end: float
    end_to_end_delta_pp: float
    divergent_steps: List[StepDivergence] = field(default_factory=list)  # ranked by |delta_pp| desc
    # Populated only in statistical detection mode; None in threshold mode.
    p_value: Optional[float] = None
    bonferroni_alpha: Optional[float] = None

    @property
    def divergence_direction(self) -> str:
        return "lower" if self.end_to_end_delta_pp < 0 else "higher"

    @property
    def is_statistically_significant(self) -> bool:
        """
        True only if p_value and bonferroni_alpha are set (statistical mode ran)
        AND the p-value is strictly below the corrected alpha. In threshold mode
        this always returns False, because significance was not tested.
        """
        return (
            self.p_value is not None
            and self.bonferroni_alpha is not None
            and self.p_value < self.bonferroni_alpha
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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
    """
    Scan each attribute column for segments with divergent end-to-end conversion.

    In ``threshold`` mode (default), a segment is returned if its end-to-end
    conversion differs from overall by at least ``min_conversion_delta``.

    In ``statistical`` mode, the same practical gate applies, AND a two-proportion
    z-test between the segment and the rest of the population must produce a
    p-value below ``alpha / N``, where ``N`` is the total number of size-passing
    segments scanned across all attribute columns in this call (Bonferroni
    correction). Segments that pass the magnitude gate but fail significance are
    dropped.

    Returned segments are sorted by absolute magnitude of end-to-end divergence,
    largest first. Direction (higher or lower than overall) is preserved.
    """
    overall = compute_funnel(events, step_order, user_col=user_col, step_col=step_col)
    overall_step_conversions = {s.step_name: s.step_conversion for s in overall.steps}

    # Successes at the final step, for the two-proportion test's oracle counts.
    overall_success = int(round(overall.total_users * overall.end_to_end_conversion))

    # First pass: gather every size-passing candidate.
    # (attr, seg_value, seg_funnel, delta)
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
            delta = seg_funnel.end_to_end_conversion - overall.end_to_end_conversion
            candidates.append((attr, seg_value, seg_funnel, delta))

    # Bonferroni denominator = number of tests we would perform. We compute
    # this over ALL size-passing candidates, not only those that also pass
    # the magnitude gate — the false-positive risk is over the whole scan.
    n_tests = max(1, len(candidates))
    bonferroni_alpha = alpha / n_tests if detection_mode == "statistical" else None

    divergent: List[DivergentSegment] = []

    for attr, seg_value, seg_funnel, delta in candidates:
        # Practical-magnitude gate applies in both modes.
        if abs(delta) < min_conversion_delta:
            continue

        p_val: Optional[float] = None
        if detection_mode == "statistical":
            seg_n = seg_funnel.total_users
            seg_success = int(round(seg_n * seg_funnel.end_to_end_conversion))
            rest_n = overall.total_users - seg_n
            rest_success = overall_success - seg_success
            p_val = _two_proportion_z_test(seg_success, seg_n, rest_success, rest_n)
            # Statistical mode's second gate.
            if p_val >= bonferroni_alpha:  # type: ignore[operator]
                continue

        # Per-segment step-level divergences (descriptive only, both modes).
        step_divs: List[StepDivergence] = []
        for step_metrics in seg_funnel.steps[:-1]:  # skip terminal
            overall_rate = overall_step_conversions.get(step_metrics.step_name)
            if overall_rate is None:
                continue
            step_delta = step_metrics.step_conversion - overall_rate
            if abs(step_delta) >= 0.02:  # ≥ 2pp step-level difference
                step_divs.append(
                    StepDivergence(
                        step_name=step_metrics.step_name,
                        segment_conversion=step_metrics.step_conversion,
                        overall_conversion=overall_rate,
                        delta_pp=step_delta,
                    )
                )
        step_divs.sort(key=lambda sd: abs(sd.delta_pp), reverse=True)

        divergent.append(
            DivergentSegment(
                segment_column=attr,
                segment_value=seg_value,
                segment_n=seg_funnel.total_users,
                segment_end_to_end=seg_funnel.end_to_end_conversion,
                overall_end_to_end=overall.end_to_end_conversion,
                end_to_end_delta_pp=delta,
                divergent_steps=step_divs,
                p_value=p_val,
                bonferroni_alpha=bonferroni_alpha,
            )
        )

    divergent.sort(key=lambda d: abs(d.end_to_end_delta_pp), reverse=True)
    return divergent
