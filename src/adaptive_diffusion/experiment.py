"""Paired interruption trials, offline selector targets, and budget accounting."""
import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_splits
from .diffusion import ACTIONS, Sampler, randn
from .metrics import energy_score, metrics
from .selector import DepthSelector, distance_action, features
from .training import file_hash, load_model, seed_all


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def trials(model, dataset, config):
    cfg = config["experiment"]
    device = next(model.parameters()).device
    sampler = Sampler(model)
    for index in range(len(dataset)):
        item = dataset[index]
        positions = item["positions"].to(device)[None]
        old_mask, new_mask = (item[k].to(device)[None] for k in ("old_mask", "new_mask"))
        old_h, new_h = (model.encoder(positions, m) for m in (old_mask, new_mask))
        for seed, fraction in itertools.product(cfg["seeds"], cfg["arrival_fractions"]):
            # All methods and budgets share the exact interrupted ensemble/noise.
            generator = torch.Generator(device="cpu" if device.type == "mps" else device).manual_seed(seed + index * 100003)
            state = sampler.interrupt(old_h, cfg["samples"], model.config["pred_len"],
                                      fraction, cfg["prefix_steps"], generator)
            noise = randn(state.x.shape, device, generator)
            for budget in cfg["budgets"]:
                yield dict(index=index, scene=item["scene"], seed=seed, fraction=fraction,
                    budget=budget, state=state, noise=noise, old_h=old_h, new_h=new_h,
                    truth=item["future"].to(device),
                    features=features(old_h, new_h, old_mask, new_mask, state.time, budget))


@torch.no_grad()
def make_labels(config, data_dir, checkpoint, output):
    seed_all(config["seed"])
    model = load_model(checkpoint, config["device"])
    splits, sampler = load_splits(data_dir), Sampler(model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        xs, ys, scenes = [], [], []
        for task in trials(model, splits[split], config):
            scores = []
            for action in ACTIONS:
                samples = sampler.update(task["state"], task["old_h"], task["new_h"],
                                         action, task["budget"], task["noise"])
                scores.append(float(energy_score(samples, task["truth"])))
            xs.append(task["features"].cpu().numpy())
            ys.append(scores)
            scenes.append(task["scene"])
        np.savez_compressed(output / f"{split}.npz", features=np.stack(xs),
                            scores=np.asarray(ys, np.float32), scene=np.asarray(scenes))
    (output / "metadata.json").write_text(json.dumps(dict(
        model_sha256=file_hash(checkpoint), experiment=config["experiment"], actions=ACTIONS,
        data_hashes={s: file_hash(Path(data_dir) / f"{s}.npz") for s in ("train", "val")}
    ), indent=2))


def calibrate(labels_dir, output):
    """Fit simple distance thresholds and the best constant action without test data."""
    labels_dir = Path(labels_dir)
    meta = json.loads((labels_dir / "metadata.json").read_text())
    with np.load(labels_dir / "train.npz", allow_pickle=False) as d:
        train = d["features"].copy()
    with np.load(labels_dir / "val.npz", allow_pickle=False) as d:
        val, scores = d["features"].copy(), d["scores"].copy()
    hidden = (train.shape[1] - 8) // 4

    def distance(x, kind):
        a, b = x[:, :hidden], x[:, hidden:2 * hidden]
        if kind == "l2":
            return np.sqrt(np.mean((b - a) ** 2, axis=-1))
        return 1 - (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-8)

    result = dict(model_sha256=meta["model_sha256"], experiment=meta["experiment"],
                  fixed_action=ACTIONS[int(scores.mean(0).argmin())])
    for kind in ("l2", "cosine"):
        quantiles = np.quantile(distance(train, kind), [.2, .4, .6, .8])
        best, selected = float("inf"), None
        for multiplier in (.25, .5, 1, 2, 4):
            thresholds = quantiles * multiplier
            choices = (distance(val, kind)[:, None] > thresholds).sum(-1)
            score = scores[np.arange(len(scores)), choices].mean()
            if score < best:
                best, selected = score, thresholds.tolist()
        result[kind] = selected
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2))


@torch.no_grad()
def evaluate(config, data_dir, checkpoint, selector_path, calibration_path, output):
    seed_all(config["seed"])
    device = torch.device(config["device"])
    model, splits = load_model(checkpoint, device), load_splits(data_dir)
    sampler = Sampler(model)
    methods = config["experiment"]["methods"]
    selector, calibration = None, None
    model_hash = file_hash(checkpoint)
    if "adaptive" in methods:
        state = torch.load(selector_path, map_location=device, weights_only=True)
        if state["model_sha256"] != model_hash:
            raise ValueError("Selector belongs to a different diffusion checkpoint")
        for key in ("samples", "prefix_steps", "arrival_fractions", "budgets"):
            if state["experiment"][key] != config["experiment"][key]:
                raise ValueError(f"Selector label protocol mismatch: {key}")
        selector = DepthSelector(state["feature_dim"]).to(device)
        selector.load_state_dict(state["selector"])
        selector.eval()
    if set(methods) & {"l2", "cosine", "fixed"}:
        calibration = json.loads(Path(calibration_path).read_text())
        if calibration["model_sha256"] != model_hash:
            raise ValueError("Calibration belongs to another checkpoint")
    unknown = set(methods) - set(ACTIONS) - {"adaptive", "l2", "cosine", "fixed", "probe"}
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = None
        for task in trials(model, splits["test"], config):
            # Rotate execution order to avoid assigning all warm-up effects to one method.
            offset = task["index"] % len(methods)
            ordered = methods[offset:] + methods[:offset]
            for method in ordered:
                sync(device)
                start = time.perf_counter()
                probe_nfe = 0
                if method in ACTIONS:
                    action = method
                elif method == "adaptive":
                    action = selector.choose(task["features"])
                elif method == "fixed":
                    action = calibration["fixed_action"]
                elif method in ("l2", "cosine"):
                    action = distance_action(task["old_h"], task["new_h"], method, calibration[method])
                else:
                    # A diagnostic heuristic, NOT a reproduction of Falcon/RDM.
                    if task["budget"] <= 2:
                        action = "continue"
                    else:
                        x = task["state"].x
                        t = x.new_full((len(x),), task["state"].time)
                        old_eps = model(x, t, task["old_h"].expand(len(x), -1))
                        new_eps = model(x, t, task["new_h"].expand(len(x), -1),
                                        task["old_h"].expand(len(x), -1))
                        relative = float((new_eps - old_eps).square().mean().sqrt() /
                                         old_eps.square().mean().sqrt().clamp_min(1e-6))
                        action = ACTIONS[sum(relative > v for v in (.1, .25, .5, 1))]
                        probe_nfe = 2
                samples = sampler.update(task["state"], task["old_h"], task["new_h"],
                    action, task["budget"] - probe_nfe, task["noise"])
                sync(device)
                elapsed = time.perf_counter() - start
                nfe = task["budget"]
                row = dict(index=task["index"], scene=task["scene"], seed=task["seed"],
                    arrival_fraction=task["fraction"], actual_noise_time=task["state"].time,
                    budget=nfe, method=method, action=action, prefix_nfe=task["state"].prefix_nfe,
                    probe_nfe=probe_nfe, update_nfe=nfe, total_nfe=nfe + task["state"].prefix_nfe,
                    denoiser_sample_evaluations=nfe * len(samples), update_seconds=elapsed,
                    **metrics(samples, task["truth"]))
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
    Path(str(output) + ".metadata.json").write_text(json.dumps(dict(config=config,
        model_sha256=model_hash, test_sha256=file_hash(Path(data_dir) / "test.npz"),
        torch_version=str(torch.__version__), device=str(device),
        selector_sha256=file_hash(selector_path) if selector is not None else None,
        calibration_sha256=file_hash(calibration_path) if calibration is not None else None,
        timing="Synchronized update only; encoder/prefix/metrics excluded; includes selection/probes",
        nfe="Calls over the entire ensemble; sample evaluations also reported"), indent=2))


def summarize(results, output, seed=42):
    """Scene-cluster bootstrap; windows/repeats of a scene are not independent."""
    with Path(results).open() as handle:
        rows = list(csv.DictReader(handle))
    groups = {}
    for row in rows:
        key = (row["method"], row["budget"], row["arrival_fraction"])
        groups.setdefault(key, {}).setdefault(row["scene"], []).append(row)
    rng = np.random.default_rng(seed)
    result = []
    for (method, budget, fraction), scenes in sorted(groups.items()):
        entry = dict(method=method, budget=int(budget), arrival_fraction=float(fraction),
                     scenes=len(scenes))
        for metric in ("energy_score", "variogram_score", "min_ade", "min_fde", "mean_ade",
                       "marginal_coverage90", "interval_width90", "diversity", "update_seconds"):
            means = np.array([np.mean([float(r[metric]) for r in rs]) for rs in scenes.values()])
            boot = rng.choice(means, size=(1000, len(means)), replace=True).mean(-1)
            lo, hi = np.quantile(boot, [.025, .975])
            entry[metric] = dict(mean=float(means.mean()), ci95=[float(lo), float(hi)])
        result.append(entry)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2))
