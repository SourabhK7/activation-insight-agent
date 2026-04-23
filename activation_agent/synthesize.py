"""
Synthetic e-commerce funnel data generator.

Produces realistic-looking event data with intentionally-planted drop-off
patterns. Planting known truth matters: it lets us verify the agent surfaces
the patterns we baked in, rather than inventing stories from noise.

Planted patterns in the default generation:
1. Mobile payment friction: mobile users convert at payment step ~30pp lower
   than desktop users, and the gap is concentrated at that specific step.
2. Paid-social acquisition mismatch: users acquired via paid_social drop off
   heavily at the product_page → add_to_cart transition, consistent with an
   ad-to-landing-page mismatch.
3. Regional checkout slowdown: users in a specific country region (coded as
   country_code='CX') see elevated abandonment at shipping info.
4. A weak weekday effect (smaller magnitude, included as noise).

These are the patterns a human DS would diagnose from the data. The agent
should surface #1 and #2 clearly, #3 if it looks at country, and should NOT
fabricate patterns beyond what the data supports.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


FUNNEL_STEPS = [
    "landing_page_view",
    "product_page_view",
    "add_to_cart",
    "shipping_info_entered",
    "payment_info_entered",
    "purchase_complete",
]


@dataclass
class UserProfile:
    """Attributes that determine a user's conversion probabilities."""
    user_id: str
    device: str           # 'desktop' | 'mobile'
    acquisition_source: str  # 'organic' | 'paid_search' | 'paid_social' | 'referral'
    country_code: str     # 'US' | 'GB' | 'CA' | 'CX' (synthetic problem region)
    signup_week: int      # 0..N, used for temporal cohorting


def _sample_user_profile(user_id: str, rng: random.Random) -> UserProfile:
    """Sample a user profile from a realistic joint distribution."""
    device = rng.choices(["desktop", "mobile"], weights=[0.45, 0.55])[0]
    acquisition_source = rng.choices(
        ["organic", "paid_search", "paid_social", "referral"],
        weights=[0.40, 0.30, 0.20, 0.10],
    )[0]
    country_code = rng.choices(
        ["US", "GB", "CA", "CX"],
        weights=[0.60, 0.15, 0.15, 0.10],
    )[0]
    signup_week = rng.randint(0, 3)  # 4-week window
    return UserProfile(
        user_id=user_id,
        device=device,
        acquisition_source=acquisition_source,
        country_code=country_code,
        signup_week=signup_week,
    )


# Per-step base conversion probabilities (given reached previous step).
# These are the baseline; planted patterns modify them for specific segments.
# Calibrated to produce ~20-25% overall end-to-end, which is typical for
# well-optimized e-commerce checkout funnels.
BASE_STEP_CONVERSION = {
    "landing_page_view": 1.00,           # all users start here
    "product_page_view": 0.75,
    "add_to_cart": 0.55,
    "shipping_info_entered": 0.80,
    "payment_info_entered": 0.85,
    "purchase_complete": 0.90,
}


def _step_probability(step: str, profile: UserProfile) -> float:
    """
    Probability of completing `step` given the user reached the previous step.
    Planted drop-off patterns applied here.
    """
    p = BASE_STEP_CONVERSION[step]

    # Planted pattern 1: mobile payment friction.
    # Mobile users have a dramatically lower completion rate at the payment step.
    if step == "payment_info_entered" and profile.device == "mobile":
        p = max(0.05, p - 0.35)

    # Planted pattern 2: paid-social users have ad-to-landing mismatch.
    # Elevated drop-off at product_page → add_to_cart.
    if step == "add_to_cart" and profile.acquisition_source == "paid_social":
        p = max(0.05, p - 0.28)

    # Planted pattern 3: regional checkout slowdown in CX.
    # Elevated abandonment at shipping info step.
    if step == "shipping_info_entered" and profile.country_code == "CX":
        p = max(0.05, p - 0.22)

    # Weak temporal effect: later signup weeks have slightly worse conversion
    # at the landing → product transition (say, because marketing spend ramped
    # up and brought in lower-quality traffic).
    if step == "product_page_view":
        p = max(0.05, p - 0.03 * profile.signup_week)

    return p


def _generate_user_journey(profile: UserProfile, start_time: datetime, rng: random.Random):
    """Simulate one user's journey through the funnel. Yields event dicts."""
    current_time = start_time
    for step in FUNNEL_STEPS:
        # Every user enters at landing_page_view unconditionally.
        if step != "landing_page_view":
            conversion_prob = _step_probability(step, profile)
            if rng.random() > conversion_prob:
                return  # user dropped off before this step
            # Simulate time between steps (seconds to minutes for early steps,
            # longer for payment).
            if step in ("payment_info_entered", "purchase_complete"):
                delay_s = rng.randint(30, 300)
            else:
                delay_s = rng.randint(5, 120)
            current_time += timedelta(seconds=delay_s)

        yield {
            "user_id": profile.user_id,
            "step_name": step,
            "timestamp": current_time.isoformat(),
            "device": profile.device,
            "acquisition_source": profile.acquisition_source,
            "country_code": profile.country_code,
            "signup_week": profile.signup_week,
        }


def generate(n_users: int, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic funnel dataset."""
    rng = random.Random(seed)
    np.random.seed(seed)

    base_time = datetime(2026, 3, 1, 0, 0, 0)
    rows = []

    for i in range(n_users):
        profile = _sample_user_profile(f"user_{i:07d}", rng)
        # Spread signups across a 4-week window; users' signup_week attr
        # matches the week their events happen in.
        signup_offset_days = profile.signup_week * 7 + rng.randint(0, 6)
        signup_offset_hours = rng.randint(0, 23)
        start_time = base_time + timedelta(days=signup_offset_days, hours=signup_offset_hours)
        rows.extend(_generate_user_journey(profile, start_time, rng))

    df = pd.DataFrame(rows)
    return df


def write(df: pd.DataFrame, path: Path) -> None:
    """Write the funnel data to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    # Quick smoke test: generate a small sample and print a step-count summary.
    df = generate(n_users=1000)
    print(df.groupby("step_name").size().reindex(FUNNEL_STEPS))
