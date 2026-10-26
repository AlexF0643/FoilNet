import numpy as np
import pytest

torch = pytest.importorskip("torch")

from foilnet.data import Dataset, random_split  # noqa: E402
from foilnet.nn.ensemble import Ensemble, train_ensemble  # noqa: E402
from foilnet.nn.train import TrainConfig, train_surrogate  # noqa: E402

FEATURES = ("a", "b", "c", "d")


def toy(n, seed, span=1.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-span, span, size=(n, 4))
    y = np.column_stack([np.sin(2 * x[:, 0]) + x[:, 1] ** 2, x[:, 2] * x[:, 3]])
    return Dataset(x, y, FEATURES, ("u", "v"), {"name": "toy"})


CONFIG = TrainConfig(hidden=(16, 16), epochs=40, batch_size=32, learning_rate=5e-3)


@pytest.fixture(scope="module")
def trained():
    train, val, test = random_split(toy(400, 0), seed=0)
    ensemble, results = train_ensemble(train, val, CONFIG, n_members=4, workers=1)
    return ensemble, results, test


def test_members_differ_only_by_seed(trained):
    ensemble, results, test = trained
    assert len(ensemble.members) == 4
    assert [r.surrogate.config.seed for r in results] == [0, 1, 2, 3]
    preds = ensemble.predict_members(test.x)
    assert preds.shape == (4, len(test), 2)
    assert not np.allclose(preds[0], preds[1])


def test_predict_returns_mean_and_sample_std(trained):
    ensemble, _, test = trained
    mean, std = ensemble.predict(test.x)
    members = ensemble.predict_members(test.x)
    assert np.allclose(mean, members.mean(axis=0))
    assert np.allclose(std, members.std(axis=0, ddof=1))
    assert np.all(std > 0)


def test_ensemble_mean_beats_the_average_member(trained):
    ensemble, _, test = trained
    mean, _ = ensemble.predict(test.x)
    members = ensemble.predict_members(test.x)
    rmse_ens = np.sqrt(np.mean((mean - test.y) ** 2, axis=0))
    rmse_members = np.sqrt(np.mean((members - test.y) ** 2, axis=1)).mean(axis=0)
    assert np.all(rmse_ens <= rmse_members)  # averaging can only help (Jensen)


def test_spread_grows_away_from_the_training_data(trained):
    ensemble, _, _ = trained
    inside = toy(300, 1, span=1.0).x
    far = toy(300, 2, span=4.0).x
    far = far[np.abs(far).max(axis=1) > 2.5]  # clearly outside the [-1, 1] training box
    assert ensemble.predict(far)[1].mean() > 2.0 * ensemble.predict(inside)[1].mean()


def test_parallel_training_matches_serial():
    train, val, test = random_split(toy(200, 3), seed=0)
    cfg = TrainConfig(hidden=(8,), epochs=15, batch_size=32, learning_rate=5e-3)
    serial, _ = train_ensemble(train, val, cfg, n_members=2, workers=1)
    parallel, _ = train_ensemble(train, val, cfg, n_members=2, workers=2)
    assert np.allclose(serial.predict(test.x)[0], parallel.predict(test.x)[0], atol=1e-5)


def test_save_and_load_round_trip(trained, tmp_path):
    ensemble, _, test = trained
    ensemble.save(tmp_path / "models" / "e.pt")
    loaded = Ensemble.load(tmp_path / "models" / "e.pt")
    assert len(loaded.members) == 4
    assert np.array_equal(loaded.predict(test.x)[0], ensemble.predict(test.x)[0])
    assert np.array_equal(loaded.predict(test.x)[1], ensemble.predict(test.x)[1])
    assert loaded.feature_names == FEATURES and loaded.target_names == ("u", "v")


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        Ensemble.load(tmp_path / "none.pt")
    torch.save({"format": 99}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="not an ensemble"):
        Ensemble.load(tmp_path / "bad.pt")


def test_an_ensemble_needs_at_least_two_members(trained):
    ensemble, _, _ = trained
    with pytest.raises(ValueError, match="at least two"):
        Ensemble(ensemble.members[:1])
    train, val, _ = random_split(toy(60, 0), seed=0)
    with pytest.raises(ValueError, match="at least 2"):
        train_ensemble(train, val, CONFIG, n_members=1)


def test_members_must_share_names():
    train, val, _ = random_split(toy(80, 0), seed=0)
    a = train_surrogate(train, val, TrainConfig(hidden=(4,), epochs=2)).surrogate
    renamed = Dataset(train.x, train.y, FEATURES, ("p", "q"), {})
    b = train_surrogate(renamed, val, TrainConfig(hidden=(4,), epochs=2)).surrogate
    with pytest.raises(ValueError, match="share"):
        Ensemble((a, b))
