from pathlib import Path

import numpy as np
import pytest

from foilnet.config import ConfigError, load_config
from foilnet.data import (
    Dataset,
    DatasetConfig,
    ShapeFamily,
    cst_feature_names,
    generate_dataset,
    generate_to_file,
    load_dataset,
    load_dataset_config,
    parse_dataset_config,
    save_dataset,
    solve_sample,
)
from foilnet.geometry import ShapeLimits, check_shape, cst_coordinates
from foilnet.solver import solve

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "dataset-cst.toml"
CST = ShapeFamily("cst", 4)


def raw_config():
    """The shipped CST config, as a dict that tests can modify."""
    return load_config(SHIPPED)


def small_config(n=40, **kw):
    cfg = load_dataset_config(SHIPPED)
    return DatasetConfig(**{**cfg.__dict__, "name": "tiny", "n_samples": n, "n_panels": 40,
                            "max_abs_cd": None, **kw})


# --- shape families -----------------------------------------------------------------------


def test_feature_names():
    assert cst_feature_names(3) == ("au0", "au1", "au2", "al0", "al1", "al2", "alpha_deg")
    assert ShapeFamily("naca4").feature_names[-1] == "alpha_deg"
    assert len(CST.feature_names) == 9


@pytest.mark.parametrize("args", [("hex",), ("cst", 1), ("cst", 11)])
def test_family_rejects_bad_arguments(args):
    with pytest.raises(ValueError):
        ShapeFamily(*args)


def test_family_builds_the_matching_contour():
    params = np.array([0.2, 0.17, 0.18, 0.12, -0.16, -0.12, -0.1, -0.02])
    assert np.array_equal(
        CST.build(params, 40), cst_coordinates(params[:4], params[4:], 40)
    )
    naca = ShapeFamily("naca4").build(np.array([0.02, 0.4, 0.12]), 40)
    assert naca.shape == (41, 2)


def test_only_cst_needs_filtering_and_has_descriptors():
    params = np.array([0.2, 0.17, 0.18, 0.12, -0.16, -0.12, -0.1, -0.02])
    assert CST.needs_filter and not ShapeFamily("naca4").needs_filter
    assert CST.is_valid(params) and not CST.is_valid(-params)
    assert set(CST.descriptors(params)) == {"thickness", "camber", "te_angle_deg"}
    assert ShapeFamily("naca4").descriptors(params[:3]) == {}


def test_family_round_trips_through_dataset_metadata():
    family = ShapeFamily("cst", 5, ShapeLimits(min_te_angle_deg=6.0))
    assert ShapeFamily.from_metadata(family.describe()) == family
    assert ShapeFamily.from_metadata({}) == ShapeFamily("naca4")  # files from before CST


# --- config -------------------------------------------------------------------------------


def test_shipped_cst_config_is_valid():
    cfg = load_dataset_config(SHIPPED)
    assert cfg.family == CST and cfg.n_samples == 10000 and cfg.n_panels == 320
    assert cfg.max_abs_cd == 0.05
    assert list(cfg.ranges) == list(CST.feature_names)
    assert cfg.family.limits.min_te_angle_deg == 5.0


def test_naca_configs_still_parse_to_the_naca_family():
    root = Path(__file__).resolve().parents[1] / "configs"
    assert load_dataset_config(root / "dataset-naca4.toml").family == ShapeFamily("naca4")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r["dataset"].update(family="bspline"), "family"),
        (lambda r: r["dataset"].update(n_coefficients=1), "n_coefficients"),
        (lambda r: r["ranges"].pop("au2"), r"ranges\.au2"),
        (lambda r: r["ranges"].update(al0=[0.1, -0.1]), "low < high"),
        (lambda r: r.setdefault("shape", {}).update(typo=1.0), "unknown key"),
        (lambda r: r["solver"].update(max_abs_cd=0.0), "max_abs_cd"),
    ],
)
def test_cst_config_rejects_bad_input(mutate, match):
    raw = raw_config()
    mutate(raw)
    with pytest.raises(ConfigError, match=match):
        parse_dataset_config(raw)


# --- generation ---------------------------------------------------------------------------


def test_every_sample_is_a_usable_shape_with_matching_labels():
    ds = generate_dataset(small_config(), workers=1)
    assert len(ds) == 40 and ds.feature_names == CST.feature_names
    for row in ds.x:
        assert not check_shape(row[:4], row[4:8], CST.limits)
    for i in (0, 20, 39):
        cl, cm, cd = solve_sample(CST, ds.x[i], 40)
        assert ds.y[i] == pytest.approx([cl, cm], abs=1e-12)
        sol = solve(cst_coordinates(ds.x[i, :4], ds.x[i, 4:8], 40), ds.x[i, 8])
        assert ds.y[i, 0] == pytest.approx(sol.cl, abs=1e-12)


def test_derived_columns_are_stored_and_respect_the_limits():
    ds = generate_dataset(small_config(), workers=1)
    assert set(ds.extras) == {"thickness", "camber", "te_angle_deg"}
    limits = CST.limits
    assert np.all(ds.extras["thickness"] >= limits.min_thickness - 1e-9)
    assert np.all(ds.extras["thickness"] <= limits.max_thickness + 1e-9)
    assert np.all(ds.extras["camber"] <= limits.max_camber + 1e-9)
    assert np.all(ds.extras["te_angle_deg"] >= limits.min_te_angle_deg - 1e-9)
    assert np.array_equal(ds.column("thickness"), ds.extras["thickness"])


def test_metadata_records_the_family_and_rejections():
    meta = generate_dataset(small_config(), workers=1).metadata
    assert meta["family"] == "cst" and meta["n_coefficients"] == 4
    assert meta["limits"]["min_te_angle_deg"] == 5.0
    assert meta["n_rejected_invalid_shape"] >= 0 and meta["n_samples"] == 40
    assert list(meta["ranges"]) == list(CST.feature_names)


def test_generation_is_reproducible_and_independent_of_workers():
    serial = generate_dataset(small_config(30), workers=1)
    parallel = generate_dataset(small_config(30), workers=2)
    assert np.array_equal(serial.x, parallel.x)
    assert np.allclose(serial.y, parallel.y, rtol=0, atol=1e-10)  # rounding only
    assert not np.array_equal(serial.x, generate_dataset(small_config(30, seed=1), workers=1).x)


def test_the_pressure_drag_safety_net_drops_failed_solves():
    everything = generate_dataset(small_config(30, max_abs_cd=None), workers=1)
    nothing = generate_dataset(small_config(30, max_abs_cd=1e-12), workers=1)
    assert len(everything) == 30 and everything.metadata["max_abs_cd_filter"] is None
    assert len(nothing) == 0 and nothing.metadata["n_dropped"] == 30
    assert nothing.metadata["max_abs_cd_filter"] == 1e-12


def test_a_range_box_with_no_usable_shapes_is_reported():
    cfg = small_config(10)
    # upper and lower coefficients both large and positive: the surfaces cross
    impossible = {k: (0.4, 0.5) for k in cfg.ranges}
    impossible["alpha_deg"] = (-5.0, 12.0)
    with pytest.raises(ConfigError, match="too few usable shapes"):
        generate_dataset(small_config(10, ranges=impossible), workers=1)


def test_naca_generation_is_unchanged_by_the_family_refactor():
    cfg = DatasetConfig(
        "tiny", 20, 4, 40,
        {
            "camber": (0.0, 0.06),
            "camber_position": (0.2, 0.6),
            "thickness": (0.06, 0.18),
            "alpha_deg": (-5.0, 12.0),
        },
    )
    ds = generate_dataset(cfg, workers=1)
    assert ds.metadata["family"] == "naca4" and ds.metadata["n_rejected_invalid_shape"] == 0
    assert ds.extras == {}
    m, p, t, alpha = ds.x[3]
    from foilnet.geometry import naca4

    assert ds.y[3, 0] == pytest.approx(solve(naca4(m, p, t, 40), alpha).cl, abs=1e-12)


# --- derived columns in the file format ---------------------------------------------------


def test_extras_survive_save_load_and_subsetting(tmp_path):
    ds = generate_to_file(small_config(24), tmp_path / "c.npz", workers=1)
    loaded = load_dataset(tmp_path / "c.npz")
    assert set(loaded.extras) == set(ds.extras)
    assert np.array_equal(loaded.extras["thickness"], ds.extras["thickness"])
    part = loaded.subset(loaded.column("thickness") > np.median(loaded.column("thickness")))
    assert len(part) == 12 and len(part.extras["camber"]) == 12


def test_column_falls_back_to_extras_and_complains_about_unknown_names():
    ds = Dataset(np.zeros((3, 2)), np.zeros((3, 1)), ("a", "b"), ("u",), {},
                 extras={"t": np.array([1.0, 2.0, 3.0])})
    assert np.array_equal(ds.column("t"), [1.0, 2.0, 3.0])
    with pytest.raises(KeyError, match="no feature or extra"):
        ds.column("nope")


def test_extras_must_match_the_number_of_rows():
    with pytest.raises(ValueError, match="extra column"):
        Dataset(np.zeros((3, 2)), np.zeros((3, 1)), ("a", "b"), ("u",), {},
                extras={"t": np.zeros(5)})


def test_files_without_extras_still_load(tmp_path):
    ds = Dataset(np.zeros((3, 2)), np.zeros((3, 1)), ("a", "b"), ("u",), {})
    save_dataset(tmp_path / "plain.npz", ds)
    assert load_dataset(tmp_path / "plain.npz").extras == {}
