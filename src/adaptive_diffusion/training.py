import hashlib
import json
from pathlib import Path
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import load_splits
from .diffusion import alpha
from .model import TrajectoryDiffusion
from .selector import DepthSelector


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TrajectoryDiffusion(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def move_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def diffusion_loss(model, batch, switch_weight):
    positions, y = batch["positions"], batch["future"]
    old_h = model.encoder(positions, batch["old_mask"])
    new_h = model.encoder(positions, batch["new_mask"])
    b = len(y)
    t = torch.rand(b, device=y.device).clamp(.001, .999)
    a = alpha(t)[:, None, None]
    noise = torch.randn_like(y)
    noisy = a.sqrt() * y + (1 - a).sqrt() * noise
    # Train both incomplete and disclosed contexts from scratch.
    use_new = torch.rand(b, 1, device=y.device) > .5
    context = torch.where(use_new, new_h, old_h)
    base = F.mse_loss(model(noisy, t, context), noise)
    if switch_weight == 0:
        return base
    # One old-context DDIM step produces a mismatched intermediate state.
    # A clean-target auxiliary loss teaches the transition-conditioned architecture
    # to repair it. This is not an epsilon target for the original noisy sample.
    s = t * .8
    sa = alpha(s)[:, None, None]
    with torch.no_grad():
        old_eps = model(noisy, t, old_h.detach())
        old_clean = (noisy - (1 - a).sqrt() * old_eps) / a.sqrt()
        interrupted = sa.sqrt() * old_clean + (1 - sa).sqrt() * old_eps
    eps = model(interrupted, s, new_h, old_h)
    reconstructed = (interrupted - (1 - sa).sqrt() * eps) / sa.sqrt()
    # SNR weighting keeps very noisy intermediate states from dominating.
    switch = ((reconstructed - y).square() * sa).mean()
    return base + switch_weight * switch


def train_diffusion(config, data_dir, output):
    seed_all(config["seed"])
    device = torch.device(config["device"])
    splits = load_splits(data_dir)
    model = TrajectoryDiffusion(**config["model"]).to(device)
    for split in splits.values():
        if split.data["positions"].shape[2] != config["model"]["obs_len"] or \
                split.data["future"].shape[1] != config["model"]["pred_len"]:
            raise ValueError("Data/model horizon mismatch")
    cfg = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loaders = {s: DataLoader(splits[s], batch_size=cfg["batch_size"], shuffle=s == "train")
               for s in ("train", "val")}
    best, history = float("inf"), []
    for epoch in range(cfg["epochs"]):
        losses = {}
        for phase, loader in loaders.items():
            model.train(phase == "train")
            total = count = 0
            # Fixed validation corruption, without consuming the training RNG stream.
            rng_context = (torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else [])
                           if phase == "val" else nullcontext())
            with rng_context:
                if phase == "val":
                    torch.manual_seed(config["seed"] + 10000)
                for batch in loader:
                    batch = move_batch(batch, device)
                    with torch.set_grad_enabled(phase == "train"):
                        loss = diffusion_loss(model, batch, cfg["switch_weight"])
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                    total += float(loss.detach()) * len(batch["future"])
                    count += len(batch["future"])
            losses[phase] = total / count
        history.append(dict(epoch=epoch, **losses))
        if losses["val"] < best:
            best = losses["val"]
            save_checkpoint(output, dict(model=model.state_dict(), model_config=model.config,
                config=config, epoch=epoch, val_loss=best,
                data_hashes={s: file_hash(Path(data_dir) / f"{s}.npz") for s in ("train", "val")}))
        print(json.dumps(history[-1]), flush=True)
    Path(str(output) + ".history.json").write_text(json.dumps(history, indent=2))


def train_selector(config, labels_dir, model_path, output):
    seed_all(config["seed"])
    labels_dir = Path(labels_dir)
    meta = json.loads((labels_dir / "metadata.json").read_text())
    if meta["model_sha256"] != file_hash(model_path):
        raise ValueError("Labels belong to a different diffusion checkpoint")
    device = torch.device(config["device"])
    sets = {}
    scenes = {}
    for split in ("train", "val"):
        with np.load(labels_dir / f"{split}.npz", allow_pickle=False) as data:
            x, y = torch.from_numpy(data["features"]), torch.from_numpy(data["scores"])
            scenes[split] = set(data["scene"].tolist())
        if not len(x) or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Invalid selector labels")
        sets[split] = TensorDataset(x, y)
    if scenes["train"] & scenes["val"]:
        raise ValueError("Selector scene leakage")
    selector = DepthSelector(sets["train"].tensors[0].shape[1]).to(device)
    x, y = (v.to(device) for v in sets["train"].tensors)
    selector.mean.copy_(x.mean(0))
    selector.scale.copy_(x.std(0, unbiased=False).clamp_min(1e-4))
    selector.score_scale.copy_(y.square().mean().sqrt().clamp_min(1e-4))
    cfg = config["selector"]
    optimizer = torch.optim.AdamW(selector.parameters(), lr=cfg["lr"])
    best = float("inf")
    for epoch in range(cfg["epochs"]):
        selector.train()
        for x, y in DataLoader(sets["train"], batch_size=cfg["batch_size"], shuffle=True):
            x, y = x.to(device), y.to(device)
            prediction = selector(x)
            loss = F.smooth_l1_loss(prediction / selector.score_scale, y / selector.score_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        selector.eval()
        with torch.no_grad():
            x, y = (v.to(device) for v in sets["val"].tensors)
            prediction = selector(x)
            regret = (y.gather(1, prediction.argmin(-1, keepdim=True)).squeeze(1) - y.min(-1).values).mean()
        if float(regret) < best:
            best = float(regret)
            save_checkpoint(output, dict(selector=selector.state_dict(),
                feature_dim=x.shape[1], model_sha256=meta["model_sha256"],
                experiment=meta["experiment"], epoch=epoch, validation_regret=best))
        print(json.dumps(dict(epoch=epoch, validation_regret=float(regret))), flush=True)
