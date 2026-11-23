from lapse.data.generate import generate_policies


def test_returns_requested_number_of_rows():
    df = generate_policies(n_policies=500, seed=1)
    assert len(df) == 500
