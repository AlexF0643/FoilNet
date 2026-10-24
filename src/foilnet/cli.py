"""Command-line entry point: ``foilnet solve | generate | train | serve``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from foilnet import __version__

# subcommand -> (help text, phase that implements it) for commands not built yet
STUBS = {
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

    generate = sub.add_parser("generate", help="generate a dataset of solver results")
    generate.add_argument(
        "--config", default="configs/dataset-naca4.toml", help="dataset config (TOML)"
    )
    generate.add_argument("--out", help="output .npz path (default: data/<dataset name>.npz)")
    generate.add_argument("--workers", type=int, help="worker processes (default: CPU count)")

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


def _run_generate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    import time

    from foilnet.config import ConfigError
    from foilnet.data import generate_to_file, load_dataset_config

    try:
        config = load_dataset_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))
    out = args.out or f"data/{config.name}.npz"
    print(f"generating {config.n_samples} samples ({config.n_panels} panels each) ...")
    start = time.perf_counter()
    dataset = generate_to_file(config, out, args.workers)
    meta = dataset.metadata
    print(f"wrote {out}: {meta['n_samples']} samples, {meta['n_dropped']} dropped")
    print(f"max |pressure drag| (numerical-error diagnostic) = {meta['max_abs_cd_pressure']:.2g}")
    print(f"took {time.perf_counter() - start:.1f} s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "solve":
        return _run_solve(args, parser)
    if args.command == "generate":
        return _run_generate(args, parser)
    phase = STUBS[args.command][1]
    msg = f"foilnet {args.command}: not implemented yet (planned for Phase {phase})"
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
