import pytest

from foilnet import __version__
from foilnet.cli import COMMANDS, main


def test_no_args_prints_help(capsys):
    assert main([]) == 0
    assert "foilnet" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize("command", COMMANDS)
def test_stubs_report_not_implemented(command, capsys):
    assert main([command]) == 2
    assert "not implemented" in capsys.readouterr().err
