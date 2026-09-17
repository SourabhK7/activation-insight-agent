"""
Anomaly decomposition.

Given a metric time series and a set of attribute columns (each row is
a metric observation tagged with attributes), decompose an observed
change in the aggregate metric into contributions from:

- Rate changes within each attribute value ("behavior")
- Mix changes across attribute values ("composition")

This is the classic same-store / rate-mix decomposition. It is the
Python-side of pattern 11 (anomaly decomposition) from
llm-ds-workflow.

Input shape: two DataFrames (or two windows of one DataFrame) with:
- attr columns (categorical)
- `n` (denominator; e.g. sessions)
- `successes` (numerator; e.g. purchases)

The aggregate rate is sum(successes) / sum(n). The decomposition
answers: how much of the change in aggregate rate is explained by
rate-per-segment moving vs. by the mix of segments shifting?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class SegmentContribution:
    segment_value: str
    before_share_of_n: float
    after_share_of_n: float
    before_rate: float
    after_rate: float
    rate_change_pp: float          # (after_rate - before_rate) * 100 (documented as pp for the LLM)
    rate_effect: float             # contribution to aggregate change from this segment's rate move
    mix_effect: float              # contribution from this segment's share-of-N change


@dataclass
class AnomalyDecomposition:
    attribute: str
    before_aggregate_rate: float
    after_aggregate_rate: float
    aggregate_delta: float          # after - before
    total_rate_effect: float        # sum of segment rate_effects
    total_mix_effect: float         # sum of segment mix_effects
    segments: List[SegmentContribution] = field(default_factory=list)


@dataclass
class AnomalyFindings:
    metric_name: str
    before_label: str               # e.g. "trailing-7-day median"
    after_label: str                # e.g. "yesterday"
    before_aggregate_rate: float
    after_aggregate_rate: float
    aggregate_delta: float
    decompositions: List[AnomalyDecomposition] = field(default_factory=list)


def _decompose_by_attr(
    before: pd.DataFrame,
    after: pd.DataFrame,
    attr: str,
    *,
    n_col: str = "n",
    succ_col: str = "successes",
) -> AnomalyDecomposition:
    """
    Rate/mix decomposition for a single attribute.

    Method:
      before_agg = sum(before.successes) / sum(before.n)
      after_agg  = sum(after.successes)  / sum(after.n)
      For each attribute value:
        rate_effect = (after_rate - before_rate) * after_share_of_n
        mix_effect  = (after_share - before_share) * before_rate
      These sum (up to a small interaction term absorbed into rate_effect
      by our choice of anchor) to the aggregate delta.

    We choose after_share for the rate_effect anchor and before_rate for
    the mix_effect anchor — this is one of two standard conventions and
    the one that reads most naturally as "the segment did X points
    worse; because it grew, that hits us Y points harder."
    """
    b = before.groupby(attr).agg(**{n_col: (n_col, "sum"), succ_col: (succ_col, "sum")})
    a = after.groupby(attr).agg(**{n_col: (n_col, "sum"), succ_col: (succ_col, "sum")})

    all_values = sorted(set(b.index) | set(a.index))
    b_total_n = b[n_col].sum()
    a_total_n = a[n_col].sum()
    b_total_s = b[succ_col].sum()
    a_total_s = a[succ_col].sum()

    before_agg = b_total_s / b_total_n if b_total_n else 0.0
    after_agg = a_total_s / a_total_n if a_total_n else 0.0
    aggregate_delta = after_agg - before_agg

    segments: List[SegmentContribution] = []
    for v in all_values:
        b_n = float(b.loc[v, n_col]) if v in b.index else 0.0
        a_n = float(a.loc[v, n_col]) if v in a.index else 0.0
        b_s = float(b.loc[v, succ_col]) if v in b.index else 0.0
        a_s = float(a.loc[v, succ_col]) if v in a.index else 0.0

        b_share = b_n / b_total_n if b_total_n else 0.0
        a_share = a_n / a_total_n if a_total_n else 0.0
        b_rate = b_s / b_n if b_n else 0.0
        a_rate = a_s / a_n if a_n else 0.0

        rate_effect = (a_rate - b_rate) * a_share
        mix_effect = (a_share - b_share) * b_rate

        segments.append(
            SegmentContribution(
                segment_value=str(v),
                before_share_of_n=b_share,
                after_share_of_n=a_share,
                before_rate=b_rate,
                after_rate=a_rate,
                rate_change_pp=(a_rate - b_rate) * 100.0,
                rate_effect=rate_effect,
                mix_effect=mix_effect,
            )
        )

    # Sort by the absolute size of total contribution (rate + mix).
    segments.sort(key=lambda s: abs(s.rate_effect + s.mix_effect), reverse=True)

    return AnomalyDecomposition(
        attribute=attr,
        before_aggregate_rate=before_agg,
        after_aggregate_rate=after_agg,
        aggregate_delta=aggregate_delta,
        total_rate_effect=sum(s.rate_effect for s in segments),
        total_mix_effect=sum(s.mix_effect for s in segments),
        segments=segments,
    )


def analyze_anomaly(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    attributes: List[str],
    metric_name: str,
    before_label: str = "before",
    after_label: str = "after",
    n_col: str = "n",
    succ_col: str = "successes",
) -> AnomalyFindings:
    """
    Produce a full anomaly-decomposition Findings object, one
    decomposition per attribute.
    """
    if before.empty or after.empty:
        raise ValueError("Both before and after DataFrames must be non-empty.")

    b_total_n = before[n_col].sum()
    a_total_n = after[n_col].sum()
    b_total_s = before[succ_col].sum()
    a_total_s = after[succ_col].sum()

    before_agg = b_total_s / b_total_n if b_total_n else 0.0
    after_agg = a_total_s / a_total_n if a_total_n else 0.0

    decomps = [
        _decompose_by_attr(before, after, attr, n_col=n_col, succ_col=succ_col)
        for attr in attributes
        if attr in before.columns and attr in after.columns
    ]

    return AnomalyFindings(
        metric_name=metric_name,
        before_label=before_label,
        after_label=after_label,
        before_aggregate_rate=before_agg,
        after_aggregate_rate=after_agg,
        aggregate_delta=after_agg - before_agg,
        decompositions=decomps,
    )
