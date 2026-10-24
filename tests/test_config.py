import pytest

from foilnet.config import ConfigError, load_config, require


def test_loads_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[a]\nb = 3\n", encoding="utf-8")
    assert load_config(p) == {"a": {"b": 3}}


def test_loads_file_with_bom(tmp_path):
    p = tmp_path / "c.toml"
    p.write_bytes(b"\xef\xbb\xbf" + b"x = 1\n")
    assert load_config(p) == {"x": 1}


def test_missing_file_is_clear(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml_names_file(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("x = = 1", encoding="utf-8")
    with pytest.raises(ConfigError, match="bad.toml"):
        load_config(p)


def test_require_names_dotted_key():
    with pytest.raises(ConfigError, match=r"solver\.n_panels"):
        require({"solver": {}}, "solver", "n_panels")
    assert require({"solver": {"n_panels": 160}}, "solver", "n_panels") == 160


def test_shipped_configs_parse():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("dataset-naca4.toml", "train-mlp.toml"):
        assert load_config(root / name)
