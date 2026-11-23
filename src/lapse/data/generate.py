"""Generate synthetic protection-insurance policy data."""

import numpy as np
import pandas as pd

PRODUCT_TYPES = ("term_life", "income_protection", "mortgage_protection")
PRODUCT_PROBS = (0.5, 0.2, 0.3)

MIN_AGE = 18
MAX_AGE = 65  # inclusive
START_DATE_MIN = pd.Timestamp("2018-01-01")
START_DATE_SPAN_DAYS = 6 * 365  # roughly 2018-2023


def generate_policies(n_policies: int, seed: int) -> pd.DataFrame:
    """Return one row per synthetic policy. Same seed -> same output.

    All distributions are invented, round-number assumptions for a portfolio
    demo. None are calibrated to any real insurer's data.
    """
    if n_policies < 1:
        raise ValueError("n_policies must be at least 1")

    rng = np.random.default_rng(seed)

    policy_id = [f"P{i:06d}" for i in range(1, n_policies + 1)]

    age_at_start = rng.integers(MIN_AGE, MAX_AGE + 1, size=n_policies)

    product_type = rng.choice(PRODUCT_TYPES, size=n_policies, p=PRODUCT_PROBS)

    annual_premium = np.round(rng.lognormal(mean=np.log(600), sigma=0.5, size=n_policies), 2)

    start_offsets = rng.integers(0, START_DATE_SPAN_DAYS, size=n_policies)
    start_date = START_DATE_MIN + pd.to_timedelta(start_offsets, unit="D")

    return pd.DataFrame(
        {
            "policy_id": policy_id,
            "age_at_start": age_at_start,
            "product_type": product_type,
            "annual_premium": annual_premium,
            "start_date": start_date,
        }
    )
