import pandas as pd
import pytest

from lapse.data.generate import MAX_AGE, MIN_AGE, PRODUCT_TYPES, generate_policies


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
    assert not df1.equals(df2)


def test_values_are_within_valid_ranges():
    df = generate_policies(n_policies=2000, seed=7)

    assert df["policy_id"].is_unique
    assert df["age_at_start"].between(MIN_AGE, MAX_AGE).all()
    assert (df["annual_premium"] > 0).all()
    assert set(df["product_type"]) <= set(PRODUCT_TYPES)


def test_rejects_non_positive_policy_count():
    with pytest.raises(ValueError):
        generate_policies(n_policies=0, seed=1)
