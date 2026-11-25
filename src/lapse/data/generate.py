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
START_DATE_SPAN_DAYS = 8 * 365  # new business written from 2018 to end of 2025

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

# Lapse simulation (invented; see docs/project-log.md, decision record).
OBSERVATION_END = pd.Timestamp("2025-12-31")  # date of the simulated data extract
DAYS_PER_MONTH = 365.25 / 12
BASE_ANNUAL_LAPSE_BY_YEAR = (0.12, 0.09, 0.06)  # policy years 1, 2, 3 onwards
FREQUENCY_LAPSE_FACTOR = {"monthly": 1.5, "annual": 1.0}
PAYMENT_LAPSE_FACTOR = {"direct_debit": 1.0, "card": 1.4}
PRODUCT_LAPSE_FACTOR = {"term_life": 1.0, "income_protection": 1.2, "mortgage_protection": 0.7}
PREMIUM_LAPSE_FACTOR_PER_DOUBLING = 1.2
PREMIUM_REFERENCE = 250.0  # premium at which the premium effect is 1.0x
BROKER_EFFECT_SIGMA = 0.15  # most brokers between about 0.75x and 1.35x
PRESSURE_SIGMA = 0.35  # hidden pressure mostly between about 0.5x and 2x

INTERNAL_PREFIX = "_"  # columns used only inside the simulation; never model features


def generate_policies(n_policies: int, seed: int) -> pd.DataFrame:
    """Return one row per synthetic policy. Same seed -> same output.

    Includes `lapse_date` (NaT if still in force at OBSERVATION_END) and the
    internal column `_latent_pressure`. Use `drop_internal_columns` before any
    modelling.
    """
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

    policies = pd.DataFrame(
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
    policies["_latent_pressure"] = rng.lognormal(mean=0.0, sigma=PRESSURE_SIGMA, size=n)
    policies["lapse_date"] = _simulate_lapse_dates(policies, rng)
    return policies


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove simulation-only columns (prefixed '_') that a model must never see."""
    return df.drop(columns=[c for c in df.columns if c.startswith(INTERNAL_PREFIX)])


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


def _simulate_lapse_dates(policies: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Simulate lapse month by month from start date to OBSERVATION_END.

    Monthly risk = baseline for the policy year x product of driver multipliers.
    Policies still in force at OBSERVATION_END get NaT (censored).
    """
    n = len(policies)
    broker_effect = dict(
        zip(
            BROKER_CODES,
            rng.lognormal(mean=0.0, sigma=BROKER_EFFECT_SIGMA, size=len(BROKER_CODES)),
            strict=True,
        )
    )
    premium_exponent = np.log2(PREMIUM_LAPSE_FACTOR_PER_DOUBLING)

    multiplier = (
        policies["premium_frequency"].map(FREQUENCY_LAPSE_FACTOR)
        * policies["payment_method"].map(PAYMENT_LAPSE_FACTOR)
        * policies["product_type"].map(PRODUCT_LAPSE_FACTOR)
        * (policies["annual_premium"] / PREMIUM_REFERENCE) ** premium_exponent
        * policies["broker_code"].map(broker_effect)
        * policies["_latent_pressure"]
    ).to_numpy()

    days_observed = (OBSERVATION_END - policies["start_date"]).dt.days.to_numpy()
    months_observed = np.floor(days_observed / DAYS_PER_MONTH).astype(int)
    max_months = max(int(months_observed.max()), 1)

    month_index = np.arange(max_months)
    policy_year = np.minimum(month_index // 12, len(BASE_ANNUAL_LAPSE_BY_YEAR) - 1)
    annual_base = np.asarray(BASE_ANNUAL_LAPSE_BY_YEAR)[policy_year]
    monthly_base = 1 - (1 - annual_base) ** (1 / 12)

    hazard = np.minimum(multiplier[:, None] * monthly_base[None, :], 1.0)
    at_risk = month_index[None, :] < months_observed[:, None]
    lapses = (rng.random((n, max_months)) < hazard) & at_risk

    has_lapsed = lapses.any(axis=1)
    first_lapse_month = lapses.argmax(axis=1)  # first True per row; 0 if none
    days_to_lapse = np.where(has_lapsed, np.floor((first_lapse_month + 1) * DAYS_PER_MONTH), np.nan)
    return policies["start_date"] + pd.to_timedelta(
        pd.Series(days_to_lapse, index=policies.index), unit="D"
    )
