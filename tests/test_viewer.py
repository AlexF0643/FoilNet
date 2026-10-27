import json
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from foilnet.cli import main  # noqa: E402
from foilnet.data import (  # noqa: E402
    DatasetConfig,
    generate_dataset,
    load_dataset_config,
    random_split,
)
from foilnet.geometry import naca4_from_code  # noqa: E402
from foilnet.nn.ensemble import train_ensemble  # noqa: E402
from foilnet.nn.train import TrainConfig, train_surrogate  # noqa: E402
from foilnet.solver import solve  # noqa: E402
from foilnet.viewer import ViewerError, ViewerModel, make_server  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TRAIN = TrainConfig(hidden=(16, 16), epochs=25, batch_size=32, learning_rate=5e-3)


def tiny_ensemble(config_name, n=120):
    cfg = load_dataset_config(CONFIGS / config_name)
    cfg = DatasetConfig(**{**cfg.__dict__, "n_samples": n, "n_panels": 40, "max_abs_cd": None})
    train, val, _ = random_split(generate_dataset(cfg, workers=1), seed=0)
    ensemble, _ = train_ensemble(train, val, TRAIN, n_members=2, workers=1)
    return ensemble, train, val


@pytest.fixture(scope="module")
def naca():
    return tiny_ensemble("dataset-naca4.toml")


@pytest.fixture(scope="module")
def cst():
    return tiny_ensemble("dataset-cst.toml")


@pytest.fixture(scope="module")
def viewer(naca):
    return ViewerModel(naca[0])


def centre(model: ViewerModel) -> dict:
    return model.info()["default"]


# --- the model description -------------------------------------------------------------------


def test_info_reads_everything_from_the_model(viewer):
    info = viewer.info()
    assert info["family"] == "naca4" and info["has_spread"] and info["n_members"] == 2
    assert info["n_panels"] == 40
    assert [f["name"] for f in info["features"]] == [
        "camber", "camber_position", "thickness", "alpha_deg"]
    for f in info["features"]:
        assert f["slider_min"] < f["train_min"] < f["train_max"] < f["slider_max"]
    json.dumps(info)  # must be serialisable


def test_presets_are_the_named_sections(viewer):
    presets = {p["name"]: p["values"] for p in viewer.info()["presets"]}
    assert presets["NACA 2412"]["camber"] == pytest.approx(0.02)
    assert presets["NACA 2412"]["thickness"] == pytest.approx(0.12)
    assert presets["NACA 0012"]["camber"] == 0.0


def test_cst_presets_are_fits_that_reproduce_the_section(cst):
    model = ViewerModel(cst[0])
    assert model.family.name == "cst" and len(model.info()["features"]) == 9
    preset = next(p for p in model.info()["presets"] if "2412" in p["name"])["values"]
    request = {**centre(model), **preset, "alpha_deg": 4.0}
    out = model.evaluate(request, sweep=False)
    reference = solve(naca4_from_code("2412", 160), 4.0)
    # a four-coefficient fit is close to, not equal to, the NACA section
    assert out["solver"]["cl"] == pytest.approx(reference.cl, abs=0.03)
    assert out["warnings"] == []


def test_a_single_network_has_no_spread(naca):
    _, train, val = naca
    single = train_surrogate(train, val, TRAIN).surrogate
    model = ViewerModel(single)
    out = model.evaluate(centre(model))
    assert out["surrogate"]["std"] is None and out["sweep"]["spread"] is None
    assert model.info()["has_spread"] is False


def test_model_without_training_information_is_refused(naca):
    _, train, val = naca
    bare = replace(train_surrogate(train, val, TRAIN).surrogate, provenance={})
    with pytest.raises(ViewerError, match="training information"):
        ViewerModel(bare)


# --- one evaluation ------------------------------------------------------------------------


def test_solver_part_matches_a_direct_solve(viewer):
    values = {**centre(viewer), "camber": 0.03, "camber_position": 0.4, "thickness": 0.12,
              "alpha_deg": 3.0}
    out = viewer.evaluate(values)
    direct = solve(naca4_from_code("3412", 40), 3.0)
    assert out["solver"]["cl"] == pytest.approx(direct.cl, abs=1e-4)
    assert out["solver"]["cm"] == pytest.approx(direct.cm, abs=1e-4)


def test_network_part_matches_the_ensemble(viewer, naca):
    values = centre(viewer)
    out = viewer.evaluate(values)
    row = np.array([[values[n] for n in viewer.feature_names]])
    mean, std = naca[0].predict(row)
    assert out["surrogate"]["mean"]["cl"] == pytest.approx(mean[0, 0])
    assert out["surrogate"]["std"]["cm"] == pytest.approx(std[0, 1])


def test_sweep_contains_the_current_angle_and_runs_beyond_training(viewer):
    values = {**centre(viewer), "alpha_deg": 5.0}
    out = viewer.evaluate(values)
    sweep = out["sweep"]
    lo, hi = viewer.ranges["alpha_deg"]
    assert sweep["alpha_deg"][0] < lo and sweep["alpha_deg"][-1] > hi
    assert sweep["train_alpha"] == [lo, hi]
    n = len(sweep["alpha_deg"])
    assert len(sweep["solver"]["cl"]) == len(sweep["surrogate"]["cl"]) == n
    assert len(sweep["spread"]["cl"]) == n
    # the solver's lift curve is straight: slope near 2 pi per radian for a thin section
    slope = np.polyfit(np.deg2rad(sweep["alpha_deg"]), sweep["solver"]["cl"], 1)[0]
    assert 5.5 < slope < 7.0
    # and the sweep agrees with the single-point answer at the nearest angle
    k = int(np.argmin(np.abs(np.array(sweep["alpha_deg"]) - 5.0)))
    near = viewer.evaluate({**values, "alpha_deg": sweep["alpha_deg"][k]}, sweep=False)
    assert sweep["solver"]["cl"][k] == pytest.approx(near["solver"]["cl"], abs=1e-4)
    assert sweep["surrogate"]["cl"][k] == pytest.approx(near["surrogate"]["mean"]["cl"], abs=1e-4)


def test_sweep_can_be_skipped(viewer):
    assert "sweep" not in viewer.evaluate(centre(viewer), sweep=False)


def test_contour_and_pressure_are_returned(viewer):
    out = viewer.evaluate(centre(viewer), sweep=False)
    assert len(out["contour"]["x"]) == 41  # 40 panels, closed contour
    cp = out["cp"]
    assert cp["upper"]["x"][0] > 0.9 and cp["lower"]["x"][-1] > 0.9  # both end at the tail
    assert len(cp["upper"]["x"]) + len(cp["lower"]["x"]) == 41  # 40 panels; the nose is shared


def test_pressure_has_no_spike_at_the_trailing_edge(viewer):
    # this section at this angle put Cp = -23 on the last panel before the trailing-edge fix
    values = {"camber": 0.05, "camber_position": 0.225, "thickness": 0.071, "alpha_deg": 11.374}
    cp = viewer.evaluate({**centre(viewer), **values}, sweep=False)["cp"]
    assert cp["upper"]["cp"][0] > 0.0 and cp["lower"]["cp"][-1] > 0.0


def test_outside_the_trained_range_is_reported(viewer):
    inside = viewer.evaluate(centre(viewer), sweep=False)
    assert inside["outside"] == [] and inside["warnings"] == []
    hi = viewer.ranges["thickness"][1]
    outside = viewer.evaluate({**centre(viewer), "thickness": hi + 0.02}, sweep=False)
    assert outside["outside"] == ["thickness"]


def test_cst_shape_limit_violations_are_reported(cst):
    model = ViewerModel(cst[0])
    values = centre(model)
    flat_nose = {**values, "al0": 0.0}  # no suction-side nose: the knife edge
    out = model.evaluate(flat_nose, sweep=False)
    assert any("shape limits" in w for w in out["warnings"])


@pytest.mark.parametrize("bad", [
    {"alpha_deg": float("nan")}, {"alpha_deg": 45.0}, {"thickness": "0.12"},
    {"thickness": True}, {"camber": 5.0}, {"thickness": None},
])
def test_bad_values_are_rejected(viewer, bad):
    with pytest.raises(ViewerError):
        viewer.evaluate({**centre(viewer), **bad})


def test_missing_value_is_rejected(viewer):
    values = centre(viewer)
    del values["camber"]
    with pytest.raises(ViewerError, match="camber"):
        viewer.evaluate(values)


# --- the HTTP layer ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def url(viewer):
    server = make_server(viewer, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def fetch(url, body=None, headers=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    hdrs = {"Content-Type": "application/json", **(headers or {})} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=hdrs or (headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=20) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read(), dict(err.headers)


def test_page_and_info_are_served(url):
    status, body, headers = fetch(url + "/")
    assert status == 200 and b"FoilNet viewer" in body
    assert "text/html" in headers["Content-Type"]
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    status, body, _ = fetch(url + "/api/info")
    assert status == 200 and json.loads(body)["family"] == "naca4"


def test_evaluate_round_trip(url, viewer):
    status, body, _ = fetch(url + "/api/evaluate", {**centre(viewer), "sweep": False})
    assert status == 200
    out = json.loads(body)
    assert "solver" in out and "sweep" not in out
    status, body, _ = fetch(url + "/api/evaluate", centre(viewer))
    assert "sweep" in json.loads(body)


def test_bad_requests_get_clear_errors(url, viewer):
    status, body, _ = fetch(url + "/api/evaluate", {**centre(viewer), "alpha_deg": 99})
    assert status == 422 and "alpha_deg" in json.loads(body)["error"]
    assert fetch(url + "/api/evaluate", raw=b"not json")[0] == 400
    assert fetch(url + "/api/evaluate", raw=b"[1, 2]")[0] == 422
    assert fetch(url + "/api/evaluate", raw=b"x" * 5000)[0] == 400
    assert fetch(url + "/api/nope")[0] == 404
    assert fetch(url + "/api/evaluate", {"a": 1}, headers={"Content-Type": "text/plain"})[0] == 415


def test_non_local_host_header_is_refused(url):
    status, body, _ = fetch(url + "/api/info", headers={"Host": "evil.example"})
    assert status == 403 and "local" in json.loads(body)["error"]
    assert fetch(url + "/api/info", headers={"Host": "localhost:8000"})[0] == 200


# --- the command ---------------------------------------------------------------------------


def test_serve_reports_a_missing_model(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--model", str(tmp_path / "nope.pt")])
    assert exc.value.code == 2
    assert "foilnet train" in capsys.readouterr().err


def test_serve_starts_and_stops(tmp_path, naca, monkeypatch, capsys):
    path = tmp_path / "e.pt"
    naca[0].save(path)

    def stop(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("http.server.HTTPServer.serve_forever", stop)
    assert main(["serve", "--model", str(path), "--port", "0"]) == 0
    out = capsys.readouterr().out
    assert "http://127.0.0.1:" in out and "stopped" in out


def test_serve_accepts_a_single_network_file(tmp_path, naca, monkeypatch, capsys):
    _, train, val = naca
    path = tmp_path / "m.pt"
    train_surrogate(train, val, TRAIN).surrogate.save(path)
    monkeypatch.setattr("http.server.HTTPServer.serve_forever",
                        lambda self, *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
    assert main(["serve", "--model", str(path), "--port", "0"]) == 0
