"""Command-line entry point: ``foilnet solve | generate | train | serve``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from foilnet import __version__

# subcommand -> (help text, phase that implements it) for commands not built yet
STUBS = {
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

    solve = sub.add_parser("solve", help="run the panel-method solver on a NACA 4-digit aerofoil")
    solve.add_argument("naca", help="NACA 4-digit code, e.g. 2412")
    solve.add_argument("--alpha", type=float, default=0.0, help="angle of attack in degrees")
    solve.add_argument("--panels", type=int, default=160, help="number of panels (even)")

    for name, (help_text, _phase) in STUBS.items():
        sub.add_parser(name, help=help_text)
    return parser


def _run_solve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from foilnet.geometry import naca4_from_code
    from foilnet.solver import solve

    try:
        solution = solve(naca4_from_code(args.naca, args.panels), args.alpha)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"NACA {args.naca}  alpha = {args.alpha:g} deg  panels = {args.panels}")
    print(f"Cl      = {solution.cl:.4f}")
    print(f"Cm(c/4) = {solution.cm:.4f}")
    print("inviscid and incompressible: no stall, no drag")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "solve":
        return _run_solve(args, parser)
    phase = STUBS[args.command][1]
    msg = f"foilnet {args.command}: not implemented yet (planned for Phase {phase})"
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
