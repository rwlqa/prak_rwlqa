"""Scene-disjoint data preparation and delayed historical observation disclosure."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def scene_split(scene):
    # Stable, independent of Python hash randomization.
    bucket = int(hashlib.sha256(str(scene).encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 7 else "val" if bucket < 9 else "test"


def masks(positions, available, rng):
    """Reveal a historical suffix and neighbors; target anchor remains available."""
    old = available.copy()
    agents, obs = old.shape
    cutoff = int(rng.integers(max(1, obs // 2), obs))
    old[:, cutoff:] = False
    old[0, -1] = available[0, -1]
    if agents > 1:
        old[1:] &= rng.random((agents - 1, 1)) > .5
    # Also randomly thin older measurements; train-time masks match deployment.
    old &= rng.random(old.shape) > .1
    old[0, -1] = True
    return old, available.copy()


def save_records(records, output, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sets = {s: [] for s in ("train", "val", "test")}
    for split, record in records:
        sets[split].append(record)
    for split, rows in sets.items():
        if not rows:
            raise ValueError(f"Empty {split} split: supply more scenes or another manifest")
    for split, rows in sets.items():
        np.savez_compressed(output / f"{split}.npz", **{
            k: np.stack([r[k] for r in rows]) for k in rows[0]
        })
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def synthetic(output, scenes=300, obs_len=8, pred_len=12, agents=4, seed=42):
    """Independent multimodal pedestrian scenes; a neighbor cues turning mode."""
    if scenes < 30 or obs_len < 2 or pred_len < 1 or agents < 2:
        raise ValueError("Need >=30 scenes, >=2 observations/agents and positive horizon")
    rng = np.random.default_rng(seed)
    records = []
    for i in range(scenes):
        speed = rng.uniform(.2, .6)
        mode = rng.choice([-1, 1])
        history = np.zeros((agents, obs_len, 2), dtype=np.float32)
        history[0, :, 0] = np.arange(-obs_len + 1, 1) * speed
        history[0, :, 1] = rng.normal(0, .03, obs_len)
        for j in range(1, agents):
            history[j, :, 0] = rng.uniform(-2, 2) + np.arange(obs_len) * rng.uniform(-.3, .3)
            history[j, :, 1] = mode * rng.uniform(.5, 2) + rng.normal(0, .1, obs_len)
        probability = .85 if mode == 1 else .15
        turn = 1 if rng.random() < probability else -1
        times = np.arange(1, pred_len + 1)
        future = np.stack((times * speed, turn * .025 * times ** 1.7), -1)
        future += rng.normal(0, .04, future.shape)
        anchor = history[0, -1].copy()
        history -= anchor
        future -= anchor
        old, new = masks(history, np.ones(history.shape[:2], bool), rng)
        records.append((scene_split(f"synthetic-{seed}-{i}"), dict(
            positions=history, old_mask=old, new_mask=new,
            future=future.astype(np.float32), scene=np.array(f"synthetic-{seed}-{i}"))))
    save_records(records, output, dict(source="synthetic", seed=seed, scenes=scenes,
                                      obs_len=obs_len, pred_len=pred_len, units="metres"))


def prepare_eth(raw, manifest_path, output, obs_len=8, pred_len=12, agents=8,
                frame_step=10, stride=10, seed=42):
    """Read frame,pedestrian,x,y text (Social-GAN/MID text export convention).

    Manifest: {"sequences": [{"file":"eth.txt", "scene":"eth", "split":"test"}, ...]}.
    Related recordings MUST share scene and split. Never split overlapping windows.
    """
    if min(frame_step, stride, pred_len, agents) < 1 or obs_len < 2:
        raise ValueError("Invalid window settings")
    specs = json.loads(Path(manifest_path).read_text())["sequences"]
    assigned, files = {}, set()
    for spec in specs:
        scene, split = spec["scene"], spec["split"]
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split {split}")
        if scene in assigned and assigned[scene] != split:
            raise ValueError(f"Scene leakage: {scene}")
        assigned[scene] = split
        file = (Path(raw) / spec["file"]).resolve()
        if file in files:
            raise ValueError(f"Duplicate recording: {file}")
        files.add(file)
    rng = np.random.default_rng(seed)
    records = []
    for spec in specs:
        rows = np.loadtxt(Path(raw) / spec["file"], ndmin=2)
        if rows.shape[1] != 4 or not np.isfinite(rows).all():
            raise ValueError("Expected four finite columns: frame pedestrian x y")
        if not np.equal(rows[:, :2], np.floor(rows[:, :2])).all():
            raise ValueError("Frame and pedestrian IDs must be integer-valued")
        table = {}
        for frame, ped, x, y in rows:
            key = (int(frame), int(ped))
            if key in table:
                raise ValueError(f"Duplicate observation {key}")
            table[key] = np.array((x, y), np.float32)
        frames = sorted(set(rows[:, 0].astype(int)))
        people = sorted(set(rows[:, 1].astype(int)))
        for start in range(frames[0], frames[-1] + 1, stride):
            obs = [start + k * frame_step for k in range(obs_len)]
            fut = [obs[-1] + k * frame_step for k in range(1, pred_len + 1)]
            # Neighbor selection depends on available history only.
            for target in people:
                if not all((f, target) in table for f in obs + fut):
                    continue
                anchor = table[obs[-1], target]
                # Fixed ID ordering avoids using withheld positions to choose top-k.
                neighbors = [p for p in people if p != target and any((f, p) in table for f in obs)]
                history = np.zeros((agents, obs_len, 2), np.float32)
                available = np.zeros((agents, obs_len), bool)
                for j, p in enumerate([target] + neighbors[:agents - 1]):
                    for k, frame in enumerate(obs):
                        if (frame, p) in table:
                            history[j, k] = table[frame, p] - anchor
                            available[j, k] = True
                old, new = masks(history, available, rng)
                records.append((spec["split"], dict(positions=history, old_mask=old,
                    new_mask=new, future=np.stack([table[f, target] - anchor for f in fut]),
                    scene=np.array(spec["scene"]))))
    save_records(records, output, dict(source="ETH/UCY text", manifest=specs,
        seed=seed, obs_len=obs_len, pred_len=pred_len, frame_step=frame_step,
        stride=stride, units="metres"))


class Trajectories(Dataset):
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.data = {k: data[k].copy() for k in
                         ("positions", "old_mask", "new_mask", "future", "scene")}
        p, old, new, y, scene = (self.data[k] for k in self.data)
        if p.ndim != 4 or p.shape[-1] != 2 or min(p.shape[:3]) < 1:
            raise ValueError("positions must have shape [N,A,O,2]")
        if old.shape != p.shape[:-1] or new.shape != old.shape:
            raise ValueError("Observation mask shape mismatch")
        if old.dtype != bool or new.dtype != bool or np.any(old & ~new):
            raise ValueError("Masks must be boolean; old mask must be a subset of new")
        if not old[:, 0, -1].all():
            raise ValueError("The target anchor must already be observed")
        if y.ndim != 3 or y.shape[0] != len(p) or y.shape[-1] != 2 or scene.shape != (len(p),):
            raise ValueError("future/scene shape mismatch")
        if not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError("Data must be finite")

    def __len__(self):
        return len(self.data["positions"])

    def __getitem__(self, index):
        return {k: str(v[index]) if k == "scene" else torch.from_numpy(v[index])
                for k, v in self.data.items()}


def load_splits(directory):
    splits = {s: Trajectories(Path(directory) / f"{s}.npz") for s in ("train", "val", "test")}
    sets = {s: set(d.data["scene"].tolist()) for s, d in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[a] & sets[b]:
            raise ValueError(f"Scene leakage between {a} and {b}")
    return splits
