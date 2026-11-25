"""Generate synthetic protection-insurance policy data.

All distributions and relationships are invented, round-number assumptions for a
portfolio demo. None are calibrated to any real insurer's data.
"""

import numpy as np
import pandas as pd

PRODUCT_TYPES = ("term_life", "income_protection", "mortgage_protection")
PRODUCT_PROBS = (0.5, 0.2, 0.3)
# Relative price per unit of cover for each product (invented).
PRODUCT_RATE_FACTOR = {"term_life": 1.0, "income_protection": 2.5, "mortgage_protection": 0.8}

MIN_AGE = 18
MAX_AGE = 65  # inclusive
TERM_OPTIONS = (10, 15, 20, 25, 30)
START_DATE_MIN = pd.Timestamp("2018-01-01")
START_DATE_SPAN_DAYS = 6 * 365  # roughly 2018-2023

PREMIUM_FREQUENCIES = ("monthly", "annual")
PREMIUM_FREQUENCY_PROBS = (0.8, 0.2)
PAYMENT_METHODS = ("direct_debit", "card")
PAYMENT_METHOD_PROBS = (0.85, 0.15)
BROKER_CODES = tuple(f"B{i:02d}" for i in range(1, 21))
REGIONS = ("dublin", "rest_of_leinster", "munster", "connacht_ulster")
REGION_PROBS = (0.30, 0.25, 0.27, 0.18)

BASE_RATE_PER_1000 = 1.2  # annual premium per 1,000 of cover at age 40 (invented)
AGE_RATE_GROWTH = 0.06  # premium rises about 6% per year of age (invented)
PREMIUM_NOISE_SIGMA = 0.15


def generate_policies(n_policies: int, seed: int) -> pd.DataFrame:
    """Return one row per synthetic policy. Same seed -> same output."""
    if n_policies < 1:
        raise ValueError("n_policies must be at least 1")

    rng = np.random.default_rng(seed)
    n = n_policies

    policy_id = [f"P{i:06d}" for i in range(1, n + 1)]
    start_offsets = rng.integers(0, START_DATE_SPAN_DAYS, size=n)
    start_date = START_DATE_MIN + pd.to_timedelta(start_offsets, unit="D")
    age_at_start = rng.integers(MIN_AGE, MAX_AGE + 1, size=n)
    product_type = rng.choice(PRODUCT_TYPES, size=n, p=PRODUCT_PROBS)
    term_years = rng.choice(TERM_OPTIONS, size=n)
    sum_assured = np.round(rng.lognormal(np.log(200_000), 0.5, size=n), -3).astype("int64")
    annual_premium = _price_premium(age_at_start, product_type, sum_assured, rng)
    premium_frequency = rng.choice(PREMIUM_FREQUENCIES, size=n, p=PREMIUM_FREQUENCY_PROBS)
    payment_method = rng.choice(PAYMENT_METHODS, size=n, p=PAYMENT_METHOD_PROBS)
    broker_code = rng.choice(BROKER_CODES, size=n)
    region = rng.choice(REGIONS, size=n, p=REGION_PROBS)  # deliberately no signal

    return pd.DataFrame(
        {
            "policy_id": policy_id,
            "start_date": start_date,
            "age_at_start": age_at_start,
            "product_type": product_type,
            "term_years": term_years,
            "sum_assured": sum_assured,
            "annual_premium": annual_premium,
            "premium_frequency": premium_frequency,
            "payment_method": payment_method,
            "broker_code": broker_code,
            "region": region,
        }
    )


def _price_premium(
    age: np.ndarray,
    product_type: np.ndarray,
    sum_assured: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Annual premium = cover x base rate x age factor x product factor x noise."""
    product_factor = pd.Series(product_type).map(PRODUCT_RATE_FACTOR).to_numpy()
    age_factor = np.exp(AGE_RATE_GROWTH * (age - 40))
    noise = rng.lognormal(mean=0.0, sigma=PREMIUM_NOISE_SIGMA, size=len(age))
    premium = sum_assured / 1000 * BASE_RATE_PER_1000 * age_factor * product_factor * noise
    return np.round(premium, 2)
