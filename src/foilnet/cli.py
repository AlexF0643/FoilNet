"""Command-line entry point: ``foilnet solve | generate | train | serve``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from foilnet import __version__

# subcommand -> (help text, phase that implements it) for commands not built yet
STUBS = {
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

    train = sub.add_parser("train", help="train the PyTorch surrogate on a generated dataset")
    train.add_argument("--config", default="configs/train-mlp.toml", help="training config (TOML)")
    train.add_argument("--data", default="data/naca4-lhs.npz", help="dataset from `generate`")
    train.add_argument("--out", default="runs/mlp.pt", help="where to save the trained model")

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


def _format_metrics(split: str, metrics: dict[str, dict[str, float]]) -> list[str]:
    lines = []
    for target, m in metrics.items():
        lines.append(
            f"{split:<5} {target:<3} rmse={m['rmse']:.2e}  mae={m['mae']:.2e}  "
            f"max|err|={m['max_abs']:.2e}  R2={m['r2']:.6f}  rmse/spread={m['rmse_over_std']:.2%}"
        )
    return lines


def _run_train(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        parser.error('training needs PyTorch: python -m pip install -e ".[ml]"')

    from foilnet.config import ConfigError
    from foilnet.data import load_dataset, random_split
    from foilnet.nn.train import evaluate, load_train_config, train_surrogate

    try:
        config = load_train_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))
    try:
        dataset = load_dataset(args.data)
    except FileNotFoundError as exc:
        parser.error(f"{exc}; create it with `foilnet generate`")
    train, val, test = random_split(dataset, config.split, seed=config.seed)
    print(f"training on {len(train)} samples, validating on {len(val)}, holding out {len(test)}")
    result = train_surrogate(train, val, config)
    result.surrogate.save(args.out)
    epochs = len(result.history["val_loss"])
    print(f"ran {epochs} epochs; restored the best one (epoch {result.best_epoch})")
    print(f"saved {args.out}")
    for split, part in (("val", val), ("test", test)):
        for line in _format_metrics(split, evaluate(result.surrogate, part)):
            print(line)
    print("the test set was used only for this report")
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
    if args.command == "train":
        return _run_train(args, parser)
    phase = STUBS[args.command][1]
    msg = f"foilnet {args.command}: not implemented yet (planned for Phase {phase})"
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
