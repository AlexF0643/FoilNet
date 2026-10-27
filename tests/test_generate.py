from pathlib import Path

import numpy as np
import pytest

from foilnet import __version__
from foilnet.config import ConfigError, load_config
from foilnet.data import (
    DatasetConfig,
    generate_dataset,
    generate_to_file,
    load_dataset,
    load_dataset_config,
    parse_dataset_config,
)
from foilnet.geometry import naca4
from foilnet.solver import SOLVER_ID, solve

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "dataset-naca4.toml"


def small_config(**overrides):
    base = dict(
        name="tiny",
        n_samples=24,
        seed=4,
        n_panels=40,
        ranges={
            "camber": (0.0, 0.06),
            "camber_position": (0.2, 0.6),
            "thickness": (0.06, 0.18),
            "alpha_deg": (-5.0, 12.0),
        },
    )
    base.update(overrides)
    return DatasetConfig(**base)


def raw_config():
    return {
        "dataset": {"name": "t", "n_samples": 10, "seed": 0},
        "solver": {"n_panels": 40},
        "ranges": {
            "camber": [0.0, 0.06],
            "camber_position": [0.2, 0.6],
            "thickness": [0.06, 0.18],
            "alpha_deg": [-5.0, 12.0],
        },
    }


def test_shipped_config_is_valid():
    cfg = load_dataset_config(SHIPPED)
    assert cfg.n_panels == 160 and cfg.n_samples == 5000
    assert cfg.ranges["alpha_deg"] == (-5.0, 12.0)


def test_labels_are_the_solver_output_for_each_row():
    ds = generate_dataset(small_config(), workers=1)
    assert len(ds) == 24
    for i in (0, 11, 23):
        m, p, t, alpha = ds.x[i]
        sol = solve(naca4(m, p, t, 40), alpha)
        assert ds.y[i] == pytest.approx([sol.cl, sol.cm], abs=1e-12)
        assert ds.cd_pressure[i] == pytest.approx(sol.cd_pressure, abs=1e-12)


def test_samples_stay_inside_the_configured_ranges():
    cfg = small_config()
    ds = generate_dataset(cfg, workers=1)
    for name, (low, high) in cfg.ranges.items():
        col = ds.column(name)
        assert col.min() >= low and col.max() <= high


def test_same_seed_same_dataset_and_workers_do_not_matter():
    cfg = small_config()
    serial = generate_dataset(cfg, workers=1)
    parallel = generate_dataset(cfg, workers=2)
    assert np.array_equal(serial.x, parallel.x)
    # labels differ only by rounding: the serial path and the workers sum in a different order
    assert np.allclose(serial.y, parallel.y, rtol=0, atol=1e-10)
    other = generate_dataset(small_config(seed=5), workers=1)
    assert not np.array_equal(serial.x, other.x)


def test_metadata_records_provenance():
    cfg = small_config()
    meta = generate_dataset(cfg, workers=1).metadata
    assert meta["solver_id"] == SOLVER_ID
    assert meta["foilnet_version"] == __version__
    assert meta["n_panels"] == 40 and meta["seed"] == 4
    assert meta["n_samples"] == 24 and meta["n_samples_requested"] == 24
    assert meta["n_dropped"] == 0
    assert meta["ranges"]["thickness"] == [0.06, 0.18]
    assert "latin hypercube" in meta["sampling"]
    for key in ("created_utc", "python", "numpy", "scipy", "solver"):
        assert meta[key]


def test_file_round_trip_keeps_provenance(tmp_path):
    cfg = small_config()
    written = generate_to_file(cfg, tmp_path / "out" / "tiny.npz", workers=1)
    loaded = load_dataset(tmp_path / "out" / "tiny.npz")
    assert np.array_equal(loaded.x, written.x) and np.array_equal(loaded.y, written.y)
    assert loaded.feature_names == ("camber", "camber_position", "thickness", "alpha_deg")
    assert loaded.target_names == ("cl", "cm")
    assert loaded.metadata["solver_id"] == SOLVER_ID


def test_lift_is_physically_sensible_across_the_dataset():
    ds = generate_dataset(small_config(n_samples=60), workers=1)
    alpha, cl = ds.column("alpha_deg"), ds.y[:, 0]
    assert np.all(np.isfinite(ds.y))
    assert np.corrcoef(alpha, cl)[0, 1] > 0.9  # lift rises with angle of attack
    assert np.all(cl[alpha > 8.0] > 0.0)


def test_config_parsing_accepts_valid_input():
    cfg = parse_dataset_config(raw_config())
    assert cfg.name == "t" and cfg.ranges["camber"] == (0.0, 0.06)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r["dataset"].pop("seed"), r"dataset\.seed"),
        (lambda r: r["solver"].pop("n_panels"), r"solver\.n_panels"),
        (lambda r: r.pop("solver"), "missing required config key: solver"),
        (lambda r: r["ranges"].pop("thickness"), r"ranges\.thickness"),
        (lambda r: r["dataset"].update(n_samples=0), "n_samples"),
        (lambda r: r["solver"].update(n_panels=7), "n_panels"),
        (lambda r: r["ranges"].update(thickness=[0.2, 0.1]), "low < high"),
        (lambda r: r["ranges"].update(thickness=[0.001, 0.1]), "thickness"),
        (lambda r: r["ranges"].update(camber=[-0.01, 0.05]), "camber"),
        (lambda r: r["ranges"].update(camber_position=[0.0, 0.5]), "camber_position"),
    ],
)
def test_config_parsing_rejects_bad_input(mutate, match):
    raw = raw_config()
    mutate(raw)
    with pytest.raises(ConfigError, match=match):
        parse_dataset_config(raw)


def test_shipped_config_matches_raw_loader():
    assert parse_dataset_config(load_config(SHIPPED)) == load_dataset_config(SHIPPED)
