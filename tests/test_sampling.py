import numpy as np
import pytest

from foilnet.data import latin_hypercube

RANGES = {"a": (0.0, 1.0), "b": (-5.0, 12.0), "c": (0.06, 0.18)}


def test_shape_and_bounds():
    x = latin_hypercube(RANGES, 200, seed=1)
    assert x.shape == (200, 3)
    for j, (low, high) in enumerate(RANGES.values()):
        assert x[:, j].min() >= low and x[:, j].max() <= high


def test_one_sample_per_bin_in_every_dimension():
    n = 50
    x = latin_hypercube(RANGES, n, seed=2)
    for j, (low, high) in enumerate(RANGES.values()):
        bins = np.floor((x[:, j] - low) / (high - low) * n).astype(int)
        assert sorted(bins) == list(range(n))


def test_same_seed_is_reproducible_and_different_seed_differs():
    a = latin_hypercube(RANGES, 30, seed=7)
    assert np.array_equal(a, latin_hypercube(RANGES, 30, seed=7))
    assert not np.array_equal(a, latin_hypercube(RANGES, 30, seed=8))


def test_column_order_follows_ranges():
    x = latin_hypercube({"first": (0.0, 1.0), "second": (100.0, 200.0)}, 20, seed=0)
    assert x[:, 0].max() <= 1.0 and x[:, 1].min() >= 100.0


@pytest.mark.parametrize(
    "ranges,n",
    [({}, 10), ({"a": (1.0, 1.0)}, 10), ({"a": (2.0, 1.0)}, 10), ({"a": (0.0, 1.0)}, 0)],
)
def test_invalid_input_rejected(ranges, n):
    with pytest.raises(ValueError):
        latin_hypercube(ranges, n, seed=0)
