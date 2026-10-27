import pytest

from foilnet import __version__
from foilnet.cli import main


def test_no_args_prints_help(capsys):
    assert main([]) == 0
    assert "foilnet" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_solve_prints_coefficients(capsys):
    assert main(["solve", "2412", "--alpha", "4"]) == 0
    out = capsys.readouterr().out
    assert "Cl      = 0.74" in out
    assert "Cm(c/4) = -0.06" in out


def test_solve_rejects_bad_code(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["solve", "24"])
    assert exc.value.code == 2
    assert "NACA 4-digit" in capsys.readouterr().err


def test_generate_writes_dataset(tmp_path, capsys):
    config = tmp_path / "tiny.toml"
    config.write_text(
        """
[dataset]
name = "tiny"
n_samples = 12
seed = 1

[solver]
n_panels = 40

[ranges]
camber = [0.0, 0.06]
camber_position = [0.2, 0.6]
thickness = [0.06, 0.18]
alpha_deg = [-5.0, 12.0]
""",
        encoding="utf-8",
    )
    out = tmp_path / "tiny.npz"
    assert main(["generate", "--config", str(config), "--out", str(out), "--workers", "1"]) == 0
    assert out.is_file()
    assert "12 samples, 0 dropped" in capsys.readouterr().out


def test_generate_reports_missing_config(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--config", str(tmp_path / "nope.toml")])
    assert exc.value.code == 2
    assert "not found" in capsys.readouterr().err
