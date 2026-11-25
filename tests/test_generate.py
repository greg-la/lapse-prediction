import pandas as pd
import pytest

from lapse.data.generate import (
    BROKER_CODES,
    MAX_AGE,
    MIN_AGE,
    PAYMENT_METHODS,
    PREMIUM_FREQUENCIES,
    PRODUCT_TYPES,
    REGIONS,
    START_DATE_MIN,
    START_DATE_SPAN_DAYS,
    TERM_OPTIONS,
    generate_policies,
)

# Written out explicitly: the schema is a contract, so the test should not
# derive it from the code it is checking.
EXPECTED_COLUMNS = [
    "policy_id",
    "start_date",
    "age_at_start",
    "product_type",
    "term_years",
    "sum_assured",
    "annual_premium",
    "premium_frequency",
    "payment_method",
    "broker_code",
    "region",
]


@pytest.fixture(scope="module")
def policies() -> pd.DataFrame:
    """One shared sample, generated once for all tests in this file."""
    return generate_policies(n_policies=5000, seed=7)


def _rate_per_1000(df: pd.DataFrame) -> pd.Series:
    return df["annual_premium"] / df["sum_assured"] * 1000


def test_returns_requested_number_of_rows():
    df = generate_policies(n_policies=500, seed=1)
    assert len(df) == 500


def test_same_seed_gives_identical_output():
    df1 = generate_policies(n_policies=200, seed=42)
    df2 = generate_policies(n_policies=200, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seed_gives_different_output():
    df1 = generate_policies(n_policies=200, seed=42)
    df2 = generate_policies(n_policies=200, seed=43)
    assert not df1["annual_premium"].equals(df2["annual_premium"])


def test_has_expected_columns_in_order(policies):
    assert list(policies.columns) == EXPECTED_COLUMNS


def test_values_are_within_valid_ranges(policies):
    df = policies
    start_date_max = START_DATE_MIN + pd.Timedelta(days=START_DATE_SPAN_DAYS)

    assert df["policy_id"].is_unique
    assert df["start_date"].between(START_DATE_MIN, start_date_max).all()
    assert df["age_at_start"].between(MIN_AGE, MAX_AGE).all()
    assert set(df["product_type"]) <= set(PRODUCT_TYPES)
    assert set(df["term_years"]) <= set(TERM_OPTIONS)
    assert (df["sum_assured"] > 0).all()
    assert (df["annual_premium"] > 0).all()
    assert set(df["premium_frequency"]) <= set(PREMIUM_FREQUENCIES)
    assert set(df["payment_method"]) <= set(PAYMENT_METHODS)
    assert set(df["broker_code"]) <= set(BROKER_CODES)
    assert set(df["region"]) <= set(REGIONS)


def test_premium_rate_increases_with_age(policies):
    rate = _rate_per_1000(policies)
    young = rate[policies["age_at_start"] < 30].median()
    old = rate[policies["age_at_start"] >= 50].median()
    assert old > 2 * young


def test_premium_rate_reflects_product_pricing(policies):
    rate = _rate_per_1000(policies).groupby(policies["product_type"]).median()
    assert rate["income_protection"] > 1.5 * rate["term_life"]
    assert rate["term_life"] > 1.1 * rate["mortgage_protection"]


def test_rejects_non_positive_policy_count():
    with pytest.raises(ValueError):
        generate_policies(n_policies=0, seed=1)
