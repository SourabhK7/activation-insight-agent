"""
A/B test analysis.

Given control and treatment observed metric values (either a binary
success rate with counts, or a continuous mean with sample size and
stddev), compute:

- Absolute and relative lift
- A confidence interval on the difference
- A two-sided p-value

...then hand a structured ABFindings object to the LLM to write the
readout. Same principle as everywhere else in this repo: math in
Python, narrative from the LLM.

We do NOT try to solve every possible A/B design here — no MDE
calculators, no sequential testing, no CUPED. This module handles the
90% case (two-arm test, one metric, ended already, want a readout).
For anything else, compute the numbers elsewhere and feed them in
via `ABFindings.from_dict(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from typing import Optional


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def _two_sided_p_from_z(z: float) -> float:
    return 2.0 * (1.0 - _norm_cdf(abs(z)))


@dataclass
class ABArm:
    label: str        # e.g. "control", "treatment"
    n: int
    successes: Optional[int] = None      # for binary metrics
    mean: Optional[float] = None         # for continuous metrics
    stddev: Optional[float] = None       # for continuous metrics; sample stddev

    @property
    def rate(self) -> Optional[float]:
        if self.successes is None or self.n == 0:
            return None
        return self.successes / self.n


@dataclass
class ABResult:
    metric_kind: str  # "binary" or "continuous"
    metric_name: str
    hypothesis: str
    control: ABArm
    treatment: ABArm
    abs_lift: float                 # treatment - control (in native units)
    rel_lift: Optional[float]       # (treatment - control) / control; None if control is 0
    ci_low: float                   # 95% CI on the ABSOLUTE lift
    ci_high: float
    p_value: float                  # two-sided
    significant_at_05: bool
    directional_interpretation: str  # "treatment better", "control better", "no effect detected"
    notes: list = field(default_factory=list)


def analyze_binary(
    *,
    control_n: int,
    control_successes: int,
    treatment_n: int,
    treatment_successes: int,
    metric_name: str,
    hypothesis: str,
    alpha: float = 0.05,
) -> ABResult:
    """
    Two-proportion z-test for a binary metric.

    Notes:
    - Uses the pooled-proportion variance for the p-value (standard
      practice) and the separate-proportion variance for the CI on
      the difference (more honest when the two arms differ materially).
    - No continuity correction; sample sizes in real A/Bs are large
      enough that it doesn't matter.
    """
    if control_n <= 0 or treatment_n <= 0:
        raise ValueError("Sample sizes must be positive.")

    p1 = control_successes / control_n
    p2 = treatment_successes / treatment_n
    abs_lift = p2 - p1
    rel_lift = abs_lift / p1 if p1 > 0 else None

    # Pooled p for the p-value.
    p_pool = (control_successes + treatment_successes) / (control_n + treatment_n)
    se_pool = sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    z = abs_lift / se_pool if se_pool > 0 else 0.0
    p_value = _two_sided_p_from_z(z)

    # Separate SE for the CI.
    se_ci = sqrt(p1 * (1 - p1) / control_n + p2 * (1 - p2) / treatment_n)
    z_crit = 1.959963984540054  # two-sided 95%
    ci_low = abs_lift - z_crit * se_ci
    ci_high = abs_lift + z_crit * se_ci

    if p_value < alpha and abs_lift > 0:
        interp = "treatment better"
    elif p_value < alpha and abs_lift < 0:
        interp = "control better"
    else:
        interp = "no effect detected"

    notes = []
    # Flag tiny-sample situations. This is honest signaling for the LLM.
    if min(control_n, treatment_n) < 100:
        notes.append("At least one arm has fewer than 100 users; results are noisy.")
    if min(control_successes, treatment_successes) < 10:
        notes.append("At least one arm has fewer than 10 successes; the normal approximation may be poor.")

    return ABResult(
        metric_kind="binary",
        metric_name=metric_name,
        hypothesis=hypothesis,
        control=ABArm(label="control", n=control_n, successes=control_successes),
        treatment=ABArm(label="treatment", n=treatment_n, successes=treatment_successes),
        abs_lift=abs_lift,
        rel_lift=rel_lift,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        significant_at_05=p_value < alpha,
        directional_interpretation=interp,
        notes=notes,
    )


def analyze_continuous(
    *,
    control_n: int,
    control_mean: float,
    control_stddev: float,
    treatment_n: int,
    treatment_mean: float,
    treatment_stddev: float,
    metric_name: str,
    hypothesis: str,
    alpha: float = 0.05,
) -> ABResult:
    """
    Welch's t-test approximation for two continuous means (using the
    normal quantile for the CI — with typical A/B sample sizes the
    difference between z and t critical values is negligible).
    """
    if control_n <= 0 or treatment_n <= 0:
        raise ValueError("Sample sizes must be positive.")
    if control_stddev < 0 or treatment_stddev < 0:
        raise ValueError("Standard deviations must be non-negative.")

    abs_lift = treatment_mean - control_mean
    rel_lift = abs_lift / control_mean if control_mean != 0 else None

    se = sqrt((control_stddev ** 2) / control_n + (treatment_stddev ** 2) / treatment_n)
    z = abs_lift / se if se > 0 else 0.0
    p_value = _two_sided_p_from_z(z)

    z_crit = 1.959963984540054
    ci_low = abs_lift - z_crit * se
    ci_high = abs_lift + z_crit * se

    if p_value < alpha and abs_lift > 0:
        interp = "treatment better"
    elif p_value < alpha and abs_lift < 0:
        interp = "control better"
    else:
        interp = "no effect detected"

    notes = []
    if min(control_n, treatment_n) < 30:
        notes.append("At least one arm has fewer than 30 samples; the normal approximation may be poor.")

    return ABResult(
        metric_kind="continuous",
        metric_name=metric_name,
        hypothesis=hypothesis,
        control=ABArm(label="control", n=control_n, mean=control_mean, stddev=control_stddev),
        treatment=ABArm(label="treatment", n=treatment_n, mean=treatment_mean, stddev=treatment_stddev),
        abs_lift=abs_lift,
        rel_lift=rel_lift,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        significant_at_05=p_value < alpha,
        directional_interpretation=interp,
        notes=notes,
    )
