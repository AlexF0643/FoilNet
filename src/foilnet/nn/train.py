"""Training, evaluation and saving of the PyTorch surrogate.

Data flow: raw features and labels are standardised with statistics from the training set
only; the network works in standardised units; :class:`Surrogate` hides that, so
``predict`` takes raw (camber, camber_position, thickness, alpha_deg) and returns raw
(cl, cm).

The validation set is used for model selection (the epoch with the lowest validation loss
is restored) and for choosing hyperparameters. The test set is only ever used for the final
report.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from foilnet.config import ConfigError, load_config, require
from foilnet.data import Dataset, Standardizer
from foilnet.nn.model import ACTIVATIONS, MLP

_FORMAT = 1
_SCHEDULES = ("cosine", "constant")


@contextmanager
def _single_thread() -> Iterator[None]:
    """These networks are tiny, so PyTorch's thread pool costs more than it saves (about 3x
    slower with 4 threads). Limit it while training and put the setting back afterwards."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@dataclass(frozen=True)
class TrainConfig:
    """Defaults are the result of the validation-set search in ``docs/hyperparameters.md``."""

    hidden: tuple[int, ...] = (128, 128)
    activation: str = "tanh"
    epochs: int = 400
    batch_size: int = 128
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    schedule: str = "cosine"  # cosine decay to 1% of the initial rate, or constant
    patience: int = 0  # stop after this many epochs without improvement; 0 disables
    seed: int = 0
    split: tuple[float, float, float] = (0.7, 0.15, 0.15)


def parse_train_config(raw: dict[str, Any]) -> TrainConfig:
    """Validate a loaded TOML dict (see ``configs/train-mlp.toml``)."""
    defaults = TrainConfig()
    train = raw.get("train", {})
    cfg = TrainConfig(
        hidden=tuple(int(h) for h in require(raw, "model", "hidden")),
        activation=str(require(raw, "model", "activation")),
        epochs=int(require(raw, "train", "epochs")),
        batch_size=int(require(raw, "train", "batch_size")),
        learning_rate=float(require(raw, "train", "learning_rate")),
        weight_decay=float(train.get("weight_decay", defaults.weight_decay)),
        schedule=str(train.get("schedule", defaults.schedule)),
        patience=int(train.get("patience", defaults.patience)),
        seed=int(require(raw, "train", "seed")),
        split=(
            float(require(raw, "split", "train")),
            float(require(raw, "split", "val")),
            float(require(raw, "split", "test")),
        ),
    )
    if not cfg.hidden or min(cfg.hidden) < 1:
        raise ConfigError(f"model.hidden must be a list of positive sizes, got {cfg.hidden}")
    if cfg.activation not in ACTIVATIONS:
        raise ConfigError(
            f"model.activation must be one of {list(ACTIVATIONS)}, got {cfg.activation!r}"
        )
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ConfigError("train.epochs and train.batch_size must be at least 1")
    if cfg.learning_rate <= 0 or cfg.weight_decay < 0:
        raise ConfigError("train.learning_rate must be > 0 and train.weight_decay >= 0")
    if cfg.schedule not in _SCHEDULES:
        raise ConfigError(
            f"train.schedule must be one of {list(_SCHEDULES)}, got {cfg.schedule!r}"
        )
    if cfg.patience < 0:
        raise ConfigError("train.patience must be non-negative")
    if min(cfg.split) < 0 or abs(sum(cfg.split) - 1.0) > 1e-9:
        raise ConfigError(f"split fractions must be non-negative and sum to 1, got {cfg.split}")
    return cfg


def load_train_config(path: str | Path) -> TrainConfig:
    return parse_train_config(load_config(path))


@dataclass
class Surrogate:
    """A trained network together with everything needed to use it on raw inputs."""

    model: MLP
    x_scaler: Standardizer
    y_scaler: Standardizer
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    config: TrainConfig
    provenance: dict[str, Any] = field(default_factory=dict)  # of the training dataset

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Raw features ``(n, n_features)`` to raw predictions ``(n, n_targets)``."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError(f"x must have shape (n, {len(self.feature_names)}), got {x.shape}")
        scaled = torch.as_tensor(self.x_scaler.transform(x), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            out = self.model(scaled).numpy().astype(float)
        return self.y_scaler.inverse(out)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = asdict(self.config)
        torch.save(
            {
                "format": _FORMAT,
                "state_dict": self.model.state_dict(),
                "sizes": list(self.model.sizes),
                "activation": self.model.activation,
                "x_mean": torch.as_tensor(self.x_scaler.mean),
                "x_std": torch.as_tensor(self.x_scaler.std),
                "y_mean": torch.as_tensor(self.y_scaler.mean),
                "y_std": torch.as_tensor(self.y_scaler.std),
                "feature_names": list(self.feature_names),
                "target_names": list(self.target_names),
                "config": {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()},
                "provenance": self.provenance,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Surrogate:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        # weights_only=True refuses to unpickle arbitrary objects, so a model file cannot run code
        blob = torch.load(path, weights_only=True)
        if blob.get("format") != _FORMAT:
            raise ValueError(f"{path} has model format {blob.get('format')}, expected {_FORMAT}")
        sizes = blob["sizes"]
        model = MLP(sizes[0], sizes[1:-1], sizes[-1], blob["activation"])
        model.load_state_dict(blob["state_dict"])
        cfg = blob["config"]
        config = TrainConfig(
            **{**cfg, "hidden": tuple(cfg["hidden"]), "split": tuple(cfg["split"])}
        )
        return cls(
            model=model,
            x_scaler=Standardizer(blob["x_mean"].numpy(), blob["x_std"].numpy()),
            y_scaler=Standardizer(blob["y_mean"].numpy(), blob["y_std"].numpy()),
            feature_names=tuple(blob["feature_names"]),
            target_names=tuple(blob["target_names"]),
            config=config,
            provenance=blob["provenance"],
        )


@dataclass
class TrainResult:
    surrogate: Surrogate
    history: dict[str, list[float]]  # train_loss, val_loss (standardised units), lr
    best_epoch: int  # 1-based epoch whose weights were restored


def _tensors(dataset: Dataset, xs: Standardizer, ys: Standardizer):
    x = torch.as_tensor(xs.transform(dataset.x), dtype=torch.float32)
    y = torch.as_tensor(ys.transform(dataset.y), dtype=torch.float32)
    return x, y


def train_surrogate(train: Dataset, val: Dataset, config: TrainConfig) -> TrainResult:
    """Train on ``train``, select the best epoch on ``val``, return the restored model."""
    with _single_thread():
        return _train(train, val, config)


def _train(train: Dataset, val: Dataset, config: TrainConfig) -> TrainResult:
    x_scaler, y_scaler = Standardizer.fit(train.x), Standardizer.fit(train.y)
    x_train, y_train = _tensors(train, x_scaler, y_scaler)
    x_val, y_val = _tensors(val, x_scaler, y_scaler)

    torch.manual_seed(config.seed)  # weight initialisation
    shuffle = torch.Generator().manual_seed(config.seed)  # batch order
    model = MLP(x_train.shape[1], config.hidden, y_train.shape[1], config.activation)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=config.epochs, eta_min=0.01 * config.learning_rate
        )
        if config.schedule == "cosine"
        else None
    )
    loss_fn = torch.nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val, best_epoch, best_state, stale = float("inf"), 0, None, 0
    n = len(x_train)
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(n, generator=shuffle)
        for start in range(0, n, config.batch_size):
            batch = order[start : start + config.batch_size]
            optimiser.zero_grad()
            loss_fn(model(x_train[batch]), y_train[batch]).backward()
            optimiser.step()
        history["lr"].append(optimiser.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            history["train_loss"].append(float(loss_fn(model(x_train), y_train)))
            val_loss = float(loss_fn(model(x_val), y_val))
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if config.patience and stale >= config.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    surrogate = Surrogate(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_names=train.feature_names,
        target_names=train.target_names,
        config=config,
        provenance=dict(train.metadata),
    )
    return TrainResult(surrogate, history, best_epoch)


def evaluate(surrogate: Surrogate, dataset: Dataset) -> dict[str, dict[str, float]]:
    """Per-target error in physical units: rmse, mae, max_abs, r2, and rmse / label spread."""
    pred = surrogate.predict(dataset.x)
    err = pred - dataset.y
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(dataset.target_names):
        y = dataset.y[:, j]
        sse, sst = float(np.sum(err[:, j] ** 2)), float(np.sum((y - y.mean()) ** 2))
        rmse = float(np.sqrt(np.mean(err[:, j] ** 2)))
        out[name] = {
            "rmse": rmse,
            "mae": float(np.mean(np.abs(err[:, j]))),
            "max_abs": float(np.max(np.abs(err[:, j]))),
            "r2": 1.0 - sse / sst if sst > 0 else float("nan"),
            "rmse_over_std": rmse / float(y.std()) if y.std() > 0 else float("nan"),
        }
    return out
