import numpy as np
import pytest

from foilnet.data import Standardizer, generate_dataset, random_split
from foilnet.data.generate import DatasetConfig
from foilnet.nn import Adam, NumpyMLP, fit, gradient_check


def toy_data(n=8, n_in=4, n_out=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n_in)), rng.normal(size=(n, n_out))


# --- construction and forward pass -----------------------------------------------------


def test_parameter_shapes_and_count():
    net = NumpyMLP([4, 6, 5, 2])
    assert [w.shape for w in net.weights] == [(4, 6), (6, 5), (5, 2)]
    assert [b.shape for b in net.biases] == [(6,), (5,), (2,)]
    assert net.n_parameters == 4 * 6 + 6 + 6 * 5 + 5 + 5 * 2 + 2
    assert len(net.parameters) == 6


def test_init_is_seeded_glorot_with_zero_biases():
    a, b, c = NumpyMLP([4, 8, 2], seed=3), NumpyMLP([4, 8, 2], seed=3), NumpyMLP([4, 8, 2], seed=4)
    assert all(np.array_equal(p, q) for p, q in zip(a.parameters, b.parameters, strict=True))
    assert not np.array_equal(a.weights[0], c.weights[0])
    assert all(np.all(bias == 0.0) for bias in a.biases)
    for w in a.weights:
        assert np.abs(w).max() <= np.sqrt(6.0 / sum(w.shape))


def test_forward_matches_hand_computation():
    net = NumpyMLP([2, 2, 1], activation="tanh")
    net.weights[0] = np.array([[1.0, -1.0], [0.5, 2.0]])
    net.biases[0] = np.array([0.1, -0.2])
    net.weights[1] = np.array([[2.0], [-1.0]])
    net.biases[1] = np.array([0.3])
    x = np.array([[1.0, 2.0]])
    h = np.tanh(np.array([1.0 * 1.0 + 2.0 * 0.5 + 0.1, -1.0 + 4.0 - 0.2]))
    expected = h[0] * 2.0 - h[1] + 0.3
    assert net.predict(x)[0, 0] == pytest.approx(expected, abs=1e-15)


def test_output_layer_is_linear():
    net = NumpyMLP([1, 4, 1], seed=0)
    out = net.predict(np.array([[1e3]]))  # tanh saturates, a linear output need not
    assert abs(out[0, 0]) > 1e-3


@pytest.mark.parametrize("sizes", [[3], [3, 0, 2], []])
def test_bad_sizes_rejected(sizes):
    with pytest.raises(ValueError):
        NumpyMLP(sizes)


def test_unknown_activation_rejected():
    with pytest.raises(ValueError, match="activation"):
        NumpyMLP([2, 2], activation="swish")


def test_shape_mismatches_are_reported():
    net = NumpyMLP([3, 4, 2])
    with pytest.raises(ValueError, match="shape"):
        net.predict(np.zeros((5, 2)))
    with pytest.raises(ValueError, match="y has shape"):
        net.loss_and_grads(np.zeros((5, 3)), np.zeros((5, 1)))


# --- backprop ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["tanh", "relu"])
@pytest.mark.parametrize("l2", [0.0, 0.1])
def test_backprop_matches_finite_differences(activation, l2):
    x, y = toy_data()
    net = NumpyMLP([4, 6, 5, 2], activation, seed=1)
    assert gradient_check(net, x, y, l2) < 1e-7


def test_gradient_check_catches_a_wrong_gradient():
    x, y = toy_data()
    net = NumpyMLP([4, 6, 2], seed=1)
    real = net.loss_and_grads

    def broken(x, y, l2=0.0):
        loss, grads = real(x, y, l2)
        grads[0] = grads[0] * 1.1  # a 10% error in one weight matrix's gradient
        return loss, grads

    net.loss_and_grads = broken
    assert gradient_check(net, x, y) > 1e-3


def test_single_layer_gradient_is_least_squares():
    x, y = toy_data(n=10, n_in=3, n_out=2)
    net = NumpyMLP([3, 2], seed=2)  # no hidden layer: out = x W + b
    _, (dw, db) = net.loss_and_grads(x, y)
    resid = net.predict(x) - y
    assert np.allclose(dw, 2.0 * x.T @ resid / resid.size)
    assert np.allclose(db, 2.0 * resid.sum(axis=0) / resid.size)


def test_loss_is_mean_squared_error_plus_weight_decay():
    x, y = toy_data()
    net = NumpyMLP([4, 5, 2], seed=0)
    mse = np.mean((net.predict(x) - y) ** 2)
    assert net.loss(x, y) == pytest.approx(mse)
    penalty = 0.5 * 0.3 * sum(np.sum(w**2) for w in net.weights)
    assert net.loss(x, y, l2=0.3) == pytest.approx(mse + penalty)
    assert net.loss_and_grads(x, y, l2=0.3)[0] == pytest.approx(mse + penalty)


# --- optimiser --------------------------------------------------------------------------


def test_adam_first_step_is_learning_rate_times_sign_of_gradient():
    p = np.array([1.0, -2.0, 0.5])
    before = p.copy()
    Adam([p], lr=0.01).step([np.array([5.0, -0.003, 100.0])])
    assert np.allclose(before - p, 0.01 * np.array([1.0, -1.0, 1.0]), atol=1e-6)


def test_adam_minimises_a_quadratic():
    p = np.array([10.0, -7.0])
    opt = Adam([p], lr=0.1)
    for _ in range(500):
        opt.step([2.0 * (p - np.array([3.0, 1.0]))])
    assert p == pytest.approx([3.0, 1.0], abs=1e-3)


# --- training ---------------------------------------------------------------------------


def smooth_problem(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 2))
    y = np.column_stack([np.sin(2 * x[:, 0]) + x[:, 1] ** 2, x[:, 0] * x[:, 1]])
    return x, y


def test_fit_learns_a_smooth_function_and_generalises():
    x, y = smooth_problem(400, 0)
    x_val, y_val = smooth_problem(200, 1)
    net = NumpyMLP([2, 24, 24, 2], seed=0)
    hist = fit(net, x, y, epochs=300, batch_size=32, lr=5e-3, x_val=x_val, y_val=y_val)
    assert hist["train_loss"][-1] < 0.1 * hist["train_loss"][0]
    assert hist["val_loss"][-1] < 5e-3
    assert len(hist["train_loss"]) == len(hist["val_loss"]) == 300


def test_fit_can_memorise_a_tiny_dataset():
    x, y = toy_data(n=12, n_in=3, n_out=1, seed=5)
    net = NumpyMLP([3, 32, 32, 1], seed=0)
    hist = fit(net, x, y, epochs=1500, batch_size=12, lr=5e-3)
    assert hist["train_loss"][-1] < 1e-4


def test_fit_is_reproducible_for_a_given_seed():
    x, y = smooth_problem(100, 0)
    runs = []
    for seed in (0, 0, 1):
        net = NumpyMLP([2, 8, 2], seed=0)
        fit(net, x, y, epochs=10, batch_size=16, seed=seed)
        runs.append(net.predict(x))
    assert np.array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])


def test_weight_decay_shrinks_the_weights():
    x, y = smooth_problem(200, 0)
    norms = []
    for l2 in (0.0, 0.05):
        net = NumpyMLP([2, 16, 2], seed=0)
        fit(net, x, y, epochs=100, batch_size=32, lr=5e-3, l2=l2)
        norms.append(sum(np.sum(w**2) for w in net.weights))
    assert norms[1] < norms[0]


def test_fit_rejects_bad_settings():
    x, y = smooth_problem(10, 0)
    with pytest.raises(ValueError):
        fit(NumpyMLP([2, 4, 2]), x, y, epochs=0)
    with pytest.raises(ValueError):
        fit(NumpyMLP([2, 4, 2]), x, y, batch_size=0)


# --- on the real problem ----------------------------------------------------------------


def test_network_learns_panel_solver_output_on_a_small_dataset():
    config = DatasetConfig(
        name="tiny",
        n_samples=500,
        seed=0,
        n_panels=60,
        ranges={
            "camber": (0.0, 0.06),
            "camber_position": (0.2, 0.6),
            "thickness": (0.06, 0.18),
            "alpha_deg": (-5.0, 12.0),
        },
    )
    train, val, _ = random_split(generate_dataset(config, workers=1), seed=0)
    sx, sy = Standardizer.fit(train.x), Standardizer.fit(train.y)
    net = NumpyMLP([4, 32, 32, 2], seed=0)
    fit(net, sx.transform(train.x), sy.transform(train.y), epochs=250, batch_size=32, lr=3e-3)
    pred = sy.inverse(net.predict(sx.transform(val.x)))
    rmse = np.sqrt(np.mean((pred - val.y) ** 2, axis=0))
    # a constant prediction would give an RMSE equal to the spread of the labels
    assert np.all(rmse < 0.05 * val.y.std(axis=0))
