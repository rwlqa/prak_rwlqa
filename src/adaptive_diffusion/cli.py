import argparse
import json
from pathlib import Path

from .data import prepare_eth, synthetic
from .experiment import calibrate, evaluate, make_labels, summarize
from .training import train_diffusion, train_selector


def read_config(path):
    config = json.loads(Path(path).read_text())
    cfg = config["experiment"]
    if cfg["samples"] < 2 or cfg["prefix_steps"] < 2:
        raise ValueError("Need >=2 ensemble samples and prefix steps")
    if not cfg["budgets"] or any(type(b) is not int or b < 1 for b in cfg["budgets"]):
        raise ValueError("Budgets must be positive integers")
    if not cfg["arrival_fractions"] or any(not 0 < f < 1 for f in cfg["arrival_fractions"]):
        raise ValueError("Arrival fractions must lie strictly between 0 and 1")
    if not cfg["seeds"] or not cfg["methods"]:
        raise ValueError("Seeds and methods cannot be empty")
    for section in ("training", "selector"):
        if min(config[section][k] for k in ("epochs", "batch_size", "lr")) <= 0:
            raise ValueError(f"Invalid {section} hyperparameters")
    if config["training"]["switch_weight"] < 0:
        raise ValueError("switch_weight must be non-negative")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Adaptive trajectory diffusion experiments")
    parser.add_argument("--config", default="configs/default.json")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("synthetic", help="Prepare synthetic trajectories")
    p.add_argument("--output", required=True)
    p.add_argument("--scenes", type=int, default=300)
    p = commands.add_parser("prepare-eth", help="Prepare scene-disjoint ETH/UCY text")
    p.add_argument("--raw", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frame-step", type=int, default=10)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--agents", type=int, default=8)
    p = commands.add_parser("train-diffusion")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("make-labels")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("train-selector")
    p.add_argument("--labels", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("calibrate")
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--selector")
    p.add_argument("--calibration")
    p.add_argument("--output", required=True)
    p = commands.add_parser("summarize")
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.command == "synthetic":
        synthetic(args.output, args.scenes, config["model"]["obs_len"],
                  config["model"]["pred_len"], seed=config["seed"])
    elif args.command == "prepare-eth":
        prepare_eth(args.raw, args.manifest, args.output, config["model"]["obs_len"],
                    config["model"]["pred_len"], args.agents, args.frame_step,
                    args.stride, config["seed"])
    elif args.command == "train-diffusion":
        train_diffusion(config, args.data, args.output)
    elif args.command == "make-labels":
        make_labels(config, args.data, args.checkpoint, args.output)
    elif args.command == "train-selector":
        train_selector(config, args.labels, args.checkpoint, args.output)
    elif args.command == "calibrate":
        calibrate(args.labels, args.output)
    elif args.command == "evaluate":
        if "adaptive" in config["experiment"]["methods"] and not args.selector:
            parser.error("adaptive requires --selector")
        if set(config["experiment"]["methods"]) & {"l2", "cosine", "fixed"} and not args.calibration:
            parser.error("distance/fixed baselines require --calibration")
        evaluate(config, args.data, args.checkpoint, args.selector, args.calibration, args.output)
    else:
        summarize(args.results, args.output, config["seed"])


if __name__ == "__main__":
    main()
