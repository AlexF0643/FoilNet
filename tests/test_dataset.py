import numpy as np
import pytest

from foilnet.data import Dataset, Standardizer, load_dataset, random_split, save_dataset


def make_dataset(n=40):
    rng = np.random.default_rng(0)
    return Dataset(
        x=rng.normal(size=(n, 3)),
        y=rng.normal(size=(n, 2)),
        feature_names=("a", "b", "c"),
        target_names=("u", "v"),
        metadata={"name": "toy", "seed": 3},
        cd_pressure=rng.normal(size=n),
    )


def test_save_and_load_round_trip(tmp_path):
    ds = make_dataset()
    path = tmp_path / "sub" / "toy.npz"  # parent directory is created
    save_dataset(path, ds)
    back = load_dataset(path)
    assert np.array_equal(back.x, ds.x) and np.array_equal(back.y, ds.y)
    assert np.array_equal(back.cd_pressure, ds.cd_pressure)
    assert back.feature_names == ds.feature_names and back.target_names == ds.target_names
    assert back.metadata["name"] == "toy" and back.metadata["seed"] == 3
    assert back.metadata["format_version"] == 1


def test_round_trip_without_diagnostic(tmp_path):
    ds = make_dataset()
    ds = Dataset(ds.x, ds.y, ds.feature_names, ds.target_names, ds.metadata)
    save_dataset(tmp_path / "d.npz", ds)
    assert load_dataset(tmp_path / "d.npz").cd_pressure is None


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.npz")


def test_load_rejects_unknown_format_version(tmp_path):
    import json

    path = tmp_path / "old.npz"
    np.savez(path, x=np.zeros((1, 1)), y=np.zeros((1, 1)),
             metadata=np.array(json.dumps({"format_version": 99})))
    with pytest.raises(ValueError, match="format"):
        load_dataset(path)


def test_dataset_validates_shapes():
    with pytest.raises(ValueError):
        Dataset(np.zeros((3, 2)), np.zeros((4, 1)), ("a", "b"), ("u",), {})
    with pytest.raises(ValueError):
        Dataset(np.zeros((3, 2)), np.zeros((3, 1)), ("a",), ("u",), {})


def test_split_sizes_are_disjoint_and_cover_everything():
    ds = make_dataset(100)
    train, val, test = random_split(ds, (0.7, 0.15, 0.15), seed=1)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    rows = [map(tuple, part.x) for part in (train, val, test)]
    seen = [r for part in rows for r in part]
    assert len(set(seen)) == 100
    assert set(seen) == set(map(tuple, ds.x))


def test_split_is_reproducible_and_keeps_rows_paired():
    ds = make_dataset(60)
    a = random_split(ds, seed=5)[0]
    b = random_split(ds, seed=5)[0]
    c = random_split(ds, seed=6)[0]
    assert np.array_equal(a.x, b.x)
    assert not np.array_equal(a.x, c.x)
    lookup = {tuple(x): tuple(y) for x, y in zip(ds.x, ds.y, strict=True)}
    for x, y in zip(a.x, a.y, strict=True):
        assert lookup[tuple(x)] == tuple(y)


@pytest.mark.parametrize("fractions", [(0.5, 0.5, 0.5), (1.2, -0.1, -0.1), (0.5, 0.5)])
def test_split_rejects_bad_fractions(fractions):
    with pytest.raises(ValueError):
        random_split(make_dataset(), fractions)


def test_subset_by_named_column_mask():
    ds = make_dataset(50)
    thin = ds.subset(ds.column("b") < 0.0)
    assert len(thin) == int((ds.x[:, 1] < 0.0).sum())
    assert np.all(thin.x[:, 1] < 0.0)
    assert len(thin.cd_pressure) == len(thin)


def test_standardizer_round_trip_and_statistics():
    values = np.random.default_rng(1).normal(5.0, 3.0, size=(500, 2))
    s = Standardizer.fit(values)
    z = s.transform(values)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(z.std(axis=0), 1.0)
    assert np.allclose(s.inverse(z), values)
    assert np.allclose(s.inverse_std(np.ones(2)), s.std)


def test_standardizer_handles_constant_column():
    values = np.column_stack([np.full(10, 3.0), np.arange(10.0)])
    s = Standardizer.fit(values)
    assert np.all(np.isfinite(s.transform(values)))
    assert np.allclose(s.transform(values)[:, 0], 0.0)
