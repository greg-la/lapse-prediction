import pandas as pd

from lapse.data.generate import generate_policies


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
