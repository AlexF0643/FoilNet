"""TOML config loading with errors that say what went wrong and where."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised when a config file is missing, unreadable or not valid TOML."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a TOML file and return it as a dict.

    The file is decoded as ``utf-8-sig`` so files saved with a BOM (the Windows
    Notepad default) load cleanly.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"could not read config file {path}: {exc}") from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc


def require(config: dict[str, Any], *keys: str) -> Any:
    """Fetch a nested value, naming the full dotted key if any part is missing."""
    node: Any = config
    for i, key in enumerate(keys):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"missing required config key: {'.'.join(keys[: i + 1])}")
        node = node[key]
    return node
