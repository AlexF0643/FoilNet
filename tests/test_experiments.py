import pytest

torch = pytest.importorskip("torch")

from foilnet.analysis.experiments import (  # noqa: E402
    format_markdown,
    main,
    run_all,
    solver_discretisation_error,
)
from foilnet.data import DatasetConfig, generate_dataset, generate_to_file  # noqa: E402
from foilnet.nn.train import TrainConfig  # noqa: E402

RANGES = {
    "camber": (0.0, 0.06),
    "camber_position": (0.2, 0.6),
    "thickness": (0.06, 0.18),
    "alpha_deg": (-5.0, 12.0),
}
FAST = TrainConfig(hidden=(8, 8), epochs=8, batch_size=32, learning_rate=5e-3)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(DatasetConfig("tiny", 300, 0, 40, RANGES), workers=1)


@pytest.fixture(scope="module")
def results(dataset):
    return run_all(
        dataset, FAST, n_members=2, sizes=(20, 60), n_fine=80, workers=1, min_steps=100
    )


def test_discretisation_error_shrinks_when_the_reference_gets_closer(dataset):
    # labels are 40-panel solves; a 60-panel reference is closer to them than a 160-panel one
    near = solver_discretisation_error(dataset, 60)
    far = solver_discretisation_error(dataset, 160)
    assert all(far[k] > near[k] > 0 for k in near)


def test_discretisation_error_does_not_depend_on_worker_count(dataset):
    serial = solver_discretisation_error(dataset, 60, workers=1)
    parallel = solver_discretisation_error(dataset, 60, workers=2)
    assert serial == pytest.approx(parallel, rel=1e-9)


def test_run_all_fills_every_section(results):
    assert results.n_train + results.n_val + results.n_test == 300
    assert set(results.evaluation) == {"cl", "cm"} == set(results.label_error)
    assert [r.n_train for r in results.curve] == [20, 60]
    assert results.label_panels == 40 and results.n_fine == 80
    assert results.extrapolation.bands[0].label.startswith("in range")
    assert results.timing.speedup_batch > 1.0
    for e in results.evaluation.values():
        assert e.ensemble_rmse > 0 and e.mean_sigma > 0 and len(e.member_rmse) == 2


def test_markdown_report_has_every_section_and_numbers(results):
    text = format_markdown(results)
    for heading in (
        "# Results",
        "## Ensemble on the held-out test set",
        "### How often the error fell inside the claimed band",
        "## Solver discretisation error of the labels",
        "## Learning curve",
        "## Extrapolation",
        "## Speed",
    ):
        assert heading in text
    assert "![" not in text  # no figure links unless figures were made
    assert "solved with 40 panels" in text and "80-panel solve" in text
    assert "nan" not in text.lower().split()


def test_markdown_links_figures_when_asked(results):
    text = format_markdown(results, figures=True)
    for name in ("learning-curve", "extrapolation", "calibration"):
        assert f"assets/{name}.png" in text


def test_command_line_writes_report_figures_and_ensemble(tmp_path):
    pytest.importorskip("matplotlib")
    data = tmp_path / "d.npz"
    generate_to_file(DatasetConfig("tiny", 240, 0, 40, RANGES), data, workers=1)
    config = tmp_path / "t.toml"
    config.write_text(
        """
[model]
hidden = [8]
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
    code = main(
        [
            "--data", str(data), "--config", str(config), "--out", str(tmp_path / "r.md"),
            "--figures", str(tmp_path / "fig"), "--ensemble-out", str(tmp_path / "e.pt"),
            "--members", "2", "--workers", "1", "--sizes", "20,40", "--fine-panels", "60",
            "--min-steps", "50",
        ]
    )
    assert code == 0
    assert "## Learning curve" in (tmp_path / "r.md").read_text(encoding="utf-8")
    assert (tmp_path / "e.pt").is_file()
    for name in ("learning-curve", "extrapolation", "calibration"):
        png = tmp_path / "fig" / f"{name}.png"
        assert png.is_file() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
