"""
Cohort / segment divergence detection.

Finds segments whose funnel behavior diverges from the overall population
in ways large enough to be interesting. The goal is NOT to flag every
statistically significant difference — it's to surface the 3-5 segments
a product DS would actually want to look at.

Design choice: simple thresholds, not a full multiple-comparison framework.
The agent's output is a starting point for human review, not a ship/don't-ship
decision. Simple thresholds are easier to explain in the output diagnosis
and harder to misuse than a p-value forest.

The precise rule for "divergent segment" (current version):
  - At least MIN_SEGMENT_SIZE users reached the first step.
  - End-to-end conversion differs from overall by at least MIN_CONVERSION_DELTA
    percentage points (default 4pp).

We also identify, per divergent segment, which step(s) explain the divergence
by comparing step-by-step conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .funnel import (
    FunnelSummary,
    compute_funnel,
    compute_funnel_by_segment,
)


MIN_SEGMENT_SIZE = 500
MIN_CONVERSION_DELTA = 0.04  # 4 percentage points


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
    divergent_steps: List[StepDivergence]  # ranked by |delta_pp| descending

    @property
    def divergence_direction(self) -> str:
        return "lower" if self.end_to_end_delta_pp < 0 else "higher"


def find_divergent_segments(
    events: pd.DataFrame,
    step_order: List[str],
    attribute_columns: List[str],
    *,
    user_col: str = "user_id",
    step_col: str = "step_name",
    min_segment_size: int = MIN_SEGMENT_SIZE,
    min_conversion_delta: float = MIN_CONVERSION_DELTA,
) -> List[DivergentSegment]:
    """
    Scan each attribute column for segments with divergent end-to-end conversion.

    Returns
    -------
    List of DivergentSegment, sorted by absolute magnitude of divergence
    (largest deltas first).
    """
    overall = compute_funnel(events, step_order, user_col=user_col, step_col=step_col)
    overall_step_conversions = {s.step_name: s.step_conversion for s in overall.steps}

    divergent: List[DivergentSegment] = []

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
            if abs(delta) < min_conversion_delta:
                continue

            # For this divergent segment, identify which step(s) differ most.
            step_divs: List[StepDivergence] = []
            for step_metrics in seg_funnel.steps[:-1]:  # skip terminal step
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
                )
            )

    divergent.sort(key=lambda d: abs(d.end_to_end_delta_pp), reverse=True)
    return divergent
