import pandas as pd
import pytest

from lapse.data.generate import (
    BROKER_CODES,
    DAYS_PER_MONTH,
    MAX_AGE,
    MIN_AGE,
    OBSERVATION_END,
    PAYMENT_METHODS,
    PREMIUM_FREQUENCIES,
    PRODUCT_TYPES,
    REGIONS,
    START_DATE_MIN,
    START_DATE_SPAN_DAYS,
    TERM_OPTIONS,
    drop_internal_columns,
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
    "_latent_pressure",
    "lapse_date",
]


@pytest.fixture(scope="module")
def policies() -> pd.DataFrame:
    """One shared sample, generated once for all tests in this file."""
    return generate_policies(n_policies=20_000, seed=7)


def _rate_per_1000(df: pd.DataFrame) -> pd.Series:
    return df["annual_premium"] / df["sum_assured"] * 1000


def _lapse_rate(df: pd.DataFrame, mask: pd.Series) -> float:
    return df.loc[mask, "lapse_date"].notna().mean()


def _annual_lapse_rate(df: pd.DataFrame, policy_year: int) -> float:
    """Share of policies in force at the start of a policy year that lapse during it.

    Only policies observed for the whole year count, so recent business does
    not dilute the rate.
    """
    year_start, year_end = 12 * policy_year, 12 * (policy_year + 1)
    months_to_lapse = (df["lapse_date"] - df["start_date"]).dt.days / DAYS_PER_MONTH
    months_observed = (OBSERVATION_END - df["start_date"]).dt.days / DAYS_PER_MONTH

    in_force_at_start = ~(months_to_lapse <= year_start)
    at_risk = in_force_at_start & (months_observed >= year_end)
    lapsed_in_year = at_risk & (months_to_lapse > year_start) & (months_to_lapse <= year_end)
    return lapsed_in_year.sum() / at_risk.sum()


# --- Shape and reproducibility ------------------------------------------------


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


def test_rejects_non_positive_policy_count():
    with pytest.raises(ValueError):
        generate_policies(n_policies=0, seed=1)


# --- Valid values ------------------------------------------------------------


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
    assert (df["_latent_pressure"] > 0).all()


def test_lapse_dates_fall_within_observation_window(policies):
    lapsed = policies[policies["lapse_date"].notna()]
    assert (lapsed["lapse_date"] > lapsed["start_date"]).all()
    assert (lapsed["lapse_date"] <= OBSERVATION_END).all()


def test_overall_lapse_rate_is_plausible(policies):
    assert 0.20 < policies["lapse_date"].notna().mean() < 0.55


# --- Pricing behaviour --------------------------------------------------------


def test_premium_rate_increases_with_age(policies):
    rate = _rate_per_1000(policies)
    young = rate[policies["age_at_start"] < 30].median()
    old = rate[policies["age_at_start"] >= 50].median()
    assert old > 2 * young


def test_premium_rate_reflects_product_pricing(policies):
    rate = _rate_per_1000(policies).groupby(policies["product_type"]).median()
    assert rate["income_protection"] > 1.5 * rate["term_life"]
    assert rate["term_life"] > 1.1 * rate["mortgage_protection"]


# --- Lapse behaviour (each margin sits well below the designed effect) -------


def test_lapse_risk_is_highest_in_first_policy_year(policies):
    assert _annual_lapse_rate(policies, 0) > 1.5 * _annual_lapse_rate(policies, 2)


def test_monthly_payers_lapse_more(policies):
    freq = policies["premium_frequency"]
    monthly = _lapse_rate(policies, freq == "monthly")
    annual = _lapse_rate(policies, freq == "annual")
    assert monthly > 1.15 * annual


def test_card_payers_lapse_more(policies):
    method = policies["payment_method"]
    card = _lapse_rate(policies, method == "card")
    direct_debit = _lapse_rate(policies, method == "direct_debit")
    assert card > 1.15 * direct_debit


def test_lapse_varies_by_product(policies):
    product = policies["product_type"]
    term = _lapse_rate(policies, product == "term_life")
    income = _lapse_rate(policies, product == "income_protection")
    mortgage = _lapse_rate(policies, product == "mortgage_protection")
    assert income > 1.15 * term
    assert term > 1.15 * mortgage


def test_higher_premiums_lapse_more_within_product(policies):
    term = policies[policies["product_type"] == "term_life"]
    low_cut, high_cut = term["annual_premium"].quantile([0.25, 0.75])
    low = _lapse_rate(term, term["annual_premium"] <= low_cut)
    high = _lapse_rate(term, term["annual_premium"] >= high_cut)
    assert high > 1.2 * low


def test_region_has_no_material_effect(policies):
    rates = policies["lapse_date"].notna().groupby(policies["region"]).mean()
    assert rates.max() < 1.15 * rates.min()


# --- Leakage guard ------------------------------------------------------------


def test_drop_internal_columns_removes_hidden_fields(policies):
    observable = drop_internal_columns(policies)
    assert "_latent_pressure" in policies.columns
    assert not any(col.startswith("_") for col in observable.columns)
    assert "lapse_date" in observable.columns
    assert len(observable) == len(policies)
