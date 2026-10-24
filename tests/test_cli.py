import pytest

from foilnet import __version__
from foilnet.cli import STUBS, main


def test_no_args_prints_help(capsys):
    assert main([]) == 0
    assert "foilnet" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize("command", STUBS)
def test_stubs_report_not_implemented(command, capsys):
    assert main([command]) == 2
    assert "not implemented" in capsys.readouterr().err


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
