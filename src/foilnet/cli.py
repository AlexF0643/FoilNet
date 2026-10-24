"""Command-line entry point: ``foilnet solve | generate | train | serve``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from foilnet import __version__

# subcommand -> (help text, phase that implements it)
COMMANDS = {
    "solve": ("run the panel-method solver on one aerofoil", 2),
    "generate": ("generate a dataset of solver results", 3),
    "train": ("train a surrogate model", 5),
    "serve": ("start the interactive viewer", 8),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foilnet",
        description="A neural-network surrogate for aerofoil aerodynamics.",
    )
    parser.add_argument("--version", action="version", version=f"foilnet {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")
    for name, (help_text, _phase) in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    phase = COMMANDS[args.command][1]
    msg = f"foilnet {args.command}: not implemented yet (planned for Phase {phase})"
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
