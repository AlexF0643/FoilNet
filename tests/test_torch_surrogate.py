from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from foilnet.cli import main  # noqa: E402
from foilnet.config import ConfigError, load_config  # noqa: E402
from foilnet.data import (  # noqa: E402
    Dataset,
    DatasetConfig,
    generate_dataset,
    generate_to_file,
    random_split,
)
from foilnet.nn import NumpyMLP  # noqa: E402
from foilnet.nn.model import MLP  # noqa: E402
from foilnet.nn.train import (  # noqa: E402
    Surrogate,
    TrainConfig,
    evaluate,
    load_train_config,
    parse_train_config,
    train_surrogate,
)

RANGES = {
    "camber": (0.0, 0.06),
    "camber_position": (0.2, 0.6),
    "thickness": (0.06, 0.18),
    "alpha_deg": (-5.0, 12.0),
}
SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "train-mlp.toml"


@pytest.fixture(scope="module")
def splits():
    ds = generate_dataset(DatasetConfig("tiny", 400, 0, 60, RANGES), workers=1)
    return random_split(ds, seed=0)


def quick(**kw):
    base = {"hidden": (16, 16), "epochs": 40, "batch_size": 32, "learning_rate": 3e-3}
    return TrainConfig(**{**base, **kw})


# --- the network, and a cross-check of the hand-derived NumPy backprop against autograd ---


def test_mlp_structure_and_init():
    net = MLP(4, (8, 8), 2)
    assert net.sizes == (4, 8, 8, 2)
    assert net(torch.zeros(5, 4)).shape == (5, 2)
    linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3
    assert all(torch.all(m.bias == 0) for m in linears)


@pytest.mark.parametrize(
    "args", [(4, (8,), 2, "swish"), (0, (8,), 2, "tanh"), (4, (0,), 2, "tanh")]
)
def test_mlp_rejects_bad_arguments(args):
    with pytest.raises(ValueError):
        MLP(*args)


def _numpy_copy(net: MLP) -> NumpyMLP:
    linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    twin = NumpyMLP(net.sizes, net.activation)
    twin.weights = [m.weight.detach().numpy().T.copy() for m in linears]  # torch stores (out, in)
    twin.biases = [m.bias.detach().numpy().copy() for m in linears]
    return twin


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_torch_and_numpy_networks_agree_on_forward_and_gradients(activation):
    torch.manual_seed(0)
    net = MLP(4, (6, 5), 2, activation).double()
    # nudge the biases off zero so every term of the gradient is exercised
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.bias, std=0.3)
    twin = _numpy_copy(net)
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(9, 4)), rng.normal(size=(9, 2))

    out = net(torch.as_tensor(x))
    assert np.allclose(out.detach().numpy(), twin.predict(x), atol=1e-12)

    loss = torch.nn.functional.mse_loss(out, torch.as_tensor(y))
    loss.backward()
    value, grads = twin.loss_and_grads(x, y)
    assert float(loss.detach()) == pytest.approx(value, abs=1e-12)
    linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    for i, m in enumerate(linears):
        assert np.allclose(m.weight.grad.numpy().T, grads[2 * i], atol=1e-10)
        assert np.allclose(m.bias.grad.numpy(), grads[2 * i + 1], atol=1e-10)


# --- training ---------------------------------------------------------------------------


def test_training_reduces_validation_loss(splits):
    train, val, _ = splits
    res = train_surrogate(train, val, quick())
    h = res.history
    assert len(h["train_loss"]) == len(h["val_loss"]) == len(h["lr"]) == 40
    assert h["val_loss"][-1] < 0.5 * h["val_loss"][0]
    assert 1 <= res.best_epoch <= 40


def test_best_epoch_weights_are_restored(splits):
    train, val, _ = splits
    res = train_surrogate(train, val, quick())
    s = res.surrogate
    scaled_err = s.y_scaler.transform(s.predict(val.x)) - s.y_scaler.transform(val.y)
    assert float(np.mean(scaled_err**2)) == pytest.approx(
        res.history["val_loss"][res.best_epoch - 1], rel=1e-3
    )


def test_same_seed_is_reproducible_and_different_seed_differs(splits):
    train, val, _ = splits
    a = train_surrogate(train, val, quick(epochs=8)).surrogate.predict(val.x)
    b = train_surrogate(train, val, quick(epochs=8)).surrogate.predict(val.x)
    c = train_surrogate(train, val, quick(epochs=8, seed=1)).surrogate.predict(val.x)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_cosine_schedule_decays_and_constant_does_not(splits):
    train, val, _ = splits
    cos = train_surrogate(train, val, quick(epochs=10)).history["lr"]
    const = train_surrogate(train, val, quick(epochs=10, schedule="constant")).history["lr"]
    assert cos[0] == pytest.approx(3e-3) and cos[-1] < 0.1 * cos[0]
    assert all(lr == pytest.approx(3e-3) for lr in const)


def test_early_stopping_halts_when_validation_cannot_improve():
    rng = np.random.default_rng(0)
    make = lambda n: Dataset(  # noqa: E731
        rng.normal(size=(n, 4)), rng.normal(size=(n, 2)), ("a", "b", "c", "d"), ("u", "v"), {}
    )
    train, val = make(200), make(100)  # labels unrelated to inputs: nothing to learn
    res = train_surrogate(train, val, quick(epochs=300, patience=5, learning_rate=1e-2))
    assert len(res.history["val_loss"]) < 300
    assert len(res.history["val_loss"]) - res.best_epoch == 5


def test_training_restores_torch_thread_setting(splits):
    train, val, _ = splits
    before = torch.get_num_threads()
    train_surrogate(train, val, quick(epochs=2))
    assert torch.get_num_threads() == before


def test_surrogate_learns_the_solver_on_a_small_dataset(splits):
    train, val, _ = splits
    res = train_surrogate(train, val, TrainConfig(hidden=(32, 32), epochs=200, batch_size=32))
    metrics = evaluate(res.surrogate, val)
    for name in ("cl", "cm"):
        assert metrics[name]["rmse_over_std"] < 0.05
        assert metrics[name]["r2"] > 0.995


# --- evaluation and saving --------------------------------------------------------------


def test_evaluate_a_perfect_predictor_and_a_biased_one():
    y = np.random.default_rng(0).normal(size=(50, 2))
    ds = Dataset(np.zeros((50, 4)), y, ("a", "b", "c", "d"), ("cl", "cm"), {})
    perfect = evaluate(SimpleNamespace(predict=lambda x: y), ds)
    assert perfect["cl"]["rmse"] == 0.0 and perfect["cl"]["r2"] == 1.0
    biased = evaluate(SimpleNamespace(predict=lambda x: y + 0.5), ds)
    assert biased["cm"]["rmse"] == pytest.approx(0.5)
    assert biased["cm"]["mae"] == pytest.approx(0.5)
    assert biased["cm"]["max_abs"] == pytest.approx(0.5)
    assert biased["cm"]["rmse_over_std"] == pytest.approx(0.5 / y[:, 1].std())


def test_evaluate_matches_manual_numbers(splits):
    train, val, _ = splits
    s = train_surrogate(train, val, quick(epochs=5)).surrogate
    err = s.predict(val.x) - val.y
    m = evaluate(s, val)
    assert m["cl"]["rmse"] == pytest.approx(np.sqrt(np.mean(err[:, 0] ** 2)))
    assert m["cm"]["max_abs"] == pytest.approx(np.abs(err[:, 1]).max())


def test_save_and_load_round_trip(splits, tmp_path):
    train, val, _ = splits
    original = train_surrogate(train, val, quick(epochs=5, hidden=(12, 7))).surrogate
    path = tmp_path / "models" / "m.pt"
    original.save(path)
    loaded = Surrogate.load(path)
    assert np.array_equal(loaded.predict(val.x), original.predict(val.x))
    assert loaded.config == original.config
    assert loaded.feature_names == train.feature_names and loaded.target_names == ("cl", "cm")
    assert loaded.provenance["solver_id"] == "linear-vortex-kutta-v1"
    assert loaded.provenance["n_panels"] == 60


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        Surrogate.load(tmp_path / "nope.pt")
    torch.save({"format": 99}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="format"):
        Surrogate.load(tmp_path / "old.pt")


def test_predict_checks_input_shape(splits):
    train, val, _ = splits
    s = train_surrogate(train, val, quick(epochs=2)).surrogate
    with pytest.raises(ValueError, match="shape"):
        s.predict(np.zeros((3, 2)))
    assert s.predict(val.x[:1]).shape == (1, 2)


# --- config -----------------------------------------------------------------------------


def raw_config():
    return {
        "model": {"hidden": [16, 16], "activation": "tanh"},
        "train": {"epochs": 10, "batch_size": 32, "learning_rate": 1e-3, "seed": 0},
        "split": {"train": 0.7, "val": 0.15, "test": 0.15},
    }


def test_shipped_config_is_valid():
    cfg = load_train_config(SHIPPED)
    assert cfg.activation in ("tanh", "relu", "silu") and cfg.epochs >= 1
    assert abs(sum(cfg.split) - 1.0) < 1e-9


def test_config_defaults_for_optional_keys():
    cfg = parse_train_config(raw_config())
    defaults = TrainConfig()
    assert cfg.hidden == (16, 16)
    assert cfg.weight_decay == defaults.weight_decay
    assert cfg.schedule == defaults.schedule and cfg.patience == defaults.patience


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r["model"].pop("hidden"), r"model\.hidden"),
        (lambda r: r["train"].pop("seed"), r"train\.seed"),
        (lambda r: r["model"].update(hidden=[]), "hidden"),
        (lambda r: r["model"].update(activation="swish"), "activation"),
        (lambda r: r["train"].update(epochs=0), "epochs"),
        (lambda r: r["train"].update(learning_rate=0), "learning_rate"),
        (lambda r: r["train"].update(schedule="linear"), "schedule"),
        (lambda r: r["train"].update(patience=-1), "patience"),
        (lambda r: r["split"].update(test=0.5), "split"),
    ],
)
def test_config_rejects_bad_input(mutate, match):
    raw = raw_config()
    mutate(raw)
    with pytest.raises(ConfigError, match=match):
        parse_train_config(raw)


def test_shipped_config_matches_raw_loader():
    assert parse_train_config(load_config(SHIPPED)) == load_train_config(SHIPPED)


# --- command line -----------------------------------------------------------------------


def test_train_command_end_to_end(tmp_path, capsys):
    data = tmp_path / "d.npz"
    generate_to_file(DatasetConfig("tiny", 120, 0, 40, RANGES), data, workers=1)
    config = tmp_path / "t.toml"
    config.write_text(
        """
[model]
hidden = [8, 8]
activation = "tanh"

[train]
epochs = 4
batch_size = 32
learning_rate = 0.003
seed = 0

[split]
train = 0.7
val = 0.15
test = 0.15
""",
        encoding="utf-8",
    )
    out = tmp_path / "runs" / "m.pt"
    code = main(["train", "--config", str(config), "--data", str(data), "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0 and out.is_file()
    assert "val   cl" in text and "test  cm" in text
    assert "the test set was used only for this report" in text
    assert Surrogate.load(out).predict(np.array([[0.02, 0.4, 0.12, 4.0]])).shape == (1, 2)


def test_train_command_ensemble(tmp_path, capsys):
    from foilnet.nn.ensemble import Ensemble

    data = tmp_path / "d.npz"
    generate_to_file(DatasetConfig("tiny", 120, 0, 40, RANGES), data, workers=1)
    config = tmp_path / "t.toml"
    config.write_text(
        """
[model]
hidden = [8]
activation = "tanh"

[train]
epochs = 3
batch_size = 32
learning_rate = 0.003
seed = 0

[split]
train = 0.7
val = 0.15
test = 0.15
""",
        encoding="utf-8",
    )
    out = tmp_path / "e.pt"
    argv = ["train", "--config", str(config), "--data", str(data), "--out", str(out)]
    assert main([*argv, "--members", "2"]) == 0
    text = capsys.readouterr().out
    assert "trained 2 networks" in text and "inside +/-1 spread" in text
    mean, std = Ensemble.load(out).predict(np.array([[0.02, 0.4, 0.12, 4.0]]))
    assert mean.shape == std.shape == (1, 2)
    with pytest.raises(SystemExit):
        main([*argv, "--members", "0"])


def test_train_command_reports_missing_dataset(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train", "--config", str(SHIPPED), "--data", str(tmp_path / "none.npz")])
    assert exc.value.code == 2
    assert "foilnet generate" in capsys.readouterr().err
