import pytest

pytest.importorskip("torch")

from foilnet.analysis.hparam_search import (  # noqa: E402
    format_markdown,
    main,
    run_grid,
    staged_search,
)
from foilnet.data import DatasetConfig, generate_dataset, generate_to_file, random_split  # noqa: E402
from foilnet.nn.train import TrainConfig  # noqa: E402

RANGES = {
    "camber": (0.0, 0.06),
    "camber_position": (0.2, 0.6),
    "thickness": (0.06, 0.18),
    "alpha_deg": (-5.0, 12.0),
}


@pytest.fixture(scope="module")
def splits():
    ds = generate_dataset(DatasetConfig("tiny", 150, 0, 40, RANGES), workers=1)
    return random_split(ds, seed=0)


def test_run_grid_sorts_best_first(splits):
    train, val, _ = splits
    configs = [
        TrainConfig(hidden=(8,), epochs=2, learning_rate=1e-5),  # barely trains
        TrainConfig(hidden=(8,), epochs=30, learning_rate=1e-2),
    ]
    rows = run_grid(train, val, configs)
    assert [r.config for r in rows] == [configs[1], configs[0]]
    assert rows[0].val_loss < rows[1].val_loss
    assert set(rows[0].val_rmse) == {"cl", "cm"} and rows[0].seconds > 0


def test_staged_search_runs_three_stages_around_the_best(splits):
    train, val, _ = splits
    stages = staged_search(
        train,
        val,
        hidden=((8,), (8, 8)),
        learning_rates=(1e-2,),
        schedules=("cosine", "constant"),
        weight_decays=(0.0,),
        epochs=(3, 6),
        stage_epochs=3,
    )
    assert len(stages) == 3
    assert [len(rows) for _, rows in stages] == [2, 2, 2]
    # each stage starts from the previous winner's architecture and learning rate
    assert stages[1][1][0].config.hidden == stages[0][1][0].config.hidden
    assert {r.config.epochs for r in stages[2][1]} == {3, 6}


def test_markdown_table_lists_every_row(splits):
    train, val, _ = splits
    rows = run_grid(train, val, [TrainConfig(hidden=(8,), epochs=2)])
    text = format_markdown([("Stage X", rows)], ("cl", "cm"))
    assert "### Stage X" in text and "val RMSE cl" in text and "| 8 |" in text


def test_command_line_writes_markdown(tmp_path, capsys, monkeypatch):
    from foilnet.analysis import hparam_search

    data = tmp_path / "d.npz"
    generate_to_file(DatasetConfig("tiny", 100, 0, 40, RANGES), data, workers=1)
    out = tmp_path / "hp.md"
    full_search = hparam_search.staged_search

    def tiny_search(train, val, workers=1):  # the real grid would take minutes
        return full_search(
            train,
            val,
            hidden=((8,),),
            learning_rates=(1e-2,),
            schedules=("cosine",),
            weight_decays=(0.0,),
            epochs=(2,),
            stage_epochs=2,
            workers=workers,
        )

    monkeypatch.setattr(hparam_search, "staged_search", tiny_search)
    assert main(["--data", str(data), "--out", str(out), "--workers", "1"]) == 0
    assert out.read_text(encoding="utf-8").startswith("### Stage 1")
    assert "best configuration" in capsys.readouterr().out
