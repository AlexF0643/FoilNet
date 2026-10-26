import numpy as np
import pytest

torch = pytest.importorskip("torch")

from foilnet.analysis import timing  # noqa: E402
from foilnet.analysis.extrapolation import run_extrapolation, split_by_feature  # noqa: E402
from foilnet.analysis.learning_curve import epochs_for, learning_curve  # noqa: E402
from foilnet.data import DatasetConfig, generate_dataset, random_split  # noqa: E402
from foilnet.nn.train import TrainConfig  # noqa: E402

RANGES = {
    "camber": (0.0, 0.06),
    "camber_position": (0.2, 0.6),
    "thickness": (0.06, 0.18),
    "alpha_deg": (-5.0, 12.0),
}
FAST = TrainConfig(hidden=(16, 16), epochs=30, batch_size=32, learning_rate=5e-3)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(DatasetConfig("tiny", 400, 0, 40, RANGES), workers=1)


# --- learning curve -----------------------------------------------------------------------


def test_epochs_are_raised_for_small_datasets_only():
    cfg = TrainConfig(epochs=400, batch_size=128)
    assert epochs_for(3500, cfg) == 400  # 28 steps/epoch * 400 = 11200 steps already
    assert epochs_for(50, cfg) == 4000  # 1 step/epoch needs 4000 epochs to reach 4000 steps
    assert epochs_for(128, cfg, min_steps=1000) == 1000
    steps = lambda n: np.ceil(n / 128) * epochs_for(n, cfg)  # noqa: E731
    assert all(steps(n) >= 4000 for n in (30, 100, 500, 1000, 3500))


def test_learning_curve_rows(dataset):
    train, val, test = random_split(dataset, seed=0)
    rows = learning_curve(
        train, val, test, [30, 120], FAST, n_members=2, workers=1, min_steps=300
    )
    assert [r.n_train for r in rows] == [30, 120]
    assert rows[0].epochs > rows[1].epochs >= FAST.epochs
    for r in rows:
        assert set(r.ensemble_rmse) == {"cl", "cm"}
        assert all(v > 0 for v in r.ensemble_rmse.values())
        assert all(v > 0 for v in r.mean_sigma.values())
        # averaging members never does worse than the average member
        assert all(r.ensemble_rmse[k] <= r.member_rmse[k] + 1e-12 for k in r.ensemble_rmse)
    assert rows[1].ensemble_rmse["cl"] < rows[0].ensemble_rmse["cl"]  # more data helps


def test_learning_curve_rejects_a_size_larger_than_the_training_set(dataset):
    train, val, test = random_split(dataset, seed=0)
    with pytest.raises(ValueError, match="exceeds"):
        learning_curve(train, val, test, [len(train) + 1], FAST, n_members=2, workers=1)


# --- extrapolation ------------------------------------------------------------------------


def test_split_keeps_the_two_sides_apart_and_loses_nothing(dataset):
    split = split_by_feature(dataset, "thickness", 0.12)
    for part in (split.train, split.val, split.test_inside):
        assert np.all(part.column("thickness") < 0.12)
    assert np.all(split.outside.column("thickness") >= 0.12)
    total = len(split.train) + len(split.val) + len(split.test_inside) + len(split.outside)
    assert total == len(dataset)
    rows = [map(tuple, p.x) for p in (split.train, split.val, split.test_inside, split.outside)]
    assert len({r for part in rows for r in part}) == len(dataset)  # no sample used twice


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_split_rejects_a_threshold_that_empties_a_side(dataset, threshold):
    with pytest.raises(ValueError, match="empty"):
        split_by_feature(dataset, "thickness", threshold)


def test_extrapolation_bands_cover_the_outside_set(dataset):
    result = run_extrapolation(dataset, FAST, n_bands=3, n_members=2, workers=1)
    split = split_by_feature(dataset, "thickness", 0.12)
    assert result.bands[0].label.startswith("in range")
    outside = result.bands[1:]
    assert 1 <= len(outside) <= 3
    assert sum(b.n for b in outside) == len(split.outside)
    assert all(b.low >= 0.12 for b in outside)
    assert [b.low for b in outside] == sorted(b.low for b in outside)
    n_samples = result.bands[0].n + sum(b.n for b in outside)
    assert len(result.values) == len(result.abs_error) == len(result.sigma) == n_samples
    for b in result.bands:
        assert set(b.rmse) == {"cl", "cm"} and all(v > 0 for v in b.rmse.values())
        assert all(-1.0 <= v <= 1.0 for v in b.spearman.values())
    assert result.n_train == len(split.train)


def test_extrapolation_trains_only_on_the_thin_side(dataset):
    result = run_extrapolation(dataset, FAST, n_members=2, workers=1)
    thin = dataset.subset(dataset.column("thickness") < 0.12)
    lo, hi = thin.column("thickness").min(), thin.column("thickness").max()
    # the scaler is fitted on training data only, so its statistics describe the thin range
    scaler = result.ensemble.members[0].x_scaler
    thickness_mean = scaler.mean[dataset.feature_names.index("thickness")]
    assert lo <= thickness_mean <= hi < 0.12


# --- timing -------------------------------------------------------------------------------


def test_timing_orders_make_sense():
    weights = np.ones((4, 2))
    result = timing.measure(lambda x: x @ weights, n_panels=40, batch=2000, repeats=6)
    assert min(result.solver_new_shape, result.solver_same_shape) > 0
    assert min(result.surrogate_batch, result.surrogate_single) > 0
    assert result.solver_same_shape < result.solver_new_shape  # factorised once, reused
    assert result.surrogate_batch < result.surrogate_single  # batching amortises overhead
    assert result.speedup_batch > 1.0
    assert result.speedup_single == pytest.approx(
        result.solver_new_shape / result.surrogate_single
    )
