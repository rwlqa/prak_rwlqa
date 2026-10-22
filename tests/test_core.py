"""Small deterministic checks of mechanics, not training/benchmark runs."""
import json
import csv

import numpy as np
import pytest
import torch
from torch import nn

from adaptive_diffusion.cli import read_config
from adaptive_diffusion.data import Trajectories, load_splits, prepare_eth
from adaptive_diffusion.diffusion import ACTIONS, Sampler, State, alpha, correction_time
from adaptive_diffusion.experiment import calibrate
from adaptive_diffusion.metrics import energy_score, metrics
from adaptive_diffusion.model import TrajectoryDiffusion
from adaptive_diffusion.selector import DepthSelector, features
from adaptive_diffusion.training import diffusion_loss, load_model, save_checkpoint
from adaptive_diffusion.training import file_hash
import adaptive_diffusion.experiment as experiment

torch.set_num_threads(1)


class ZeroDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t, context, old_context=None):
        self.calls += 1
        return torch.zeros_like(x)


def test_noise_levels_are_ordered():
    for time in (.01, .1, .5, .9, .99):
        levels = [correction_time(time, a) for a in ACTIONS]
        assert levels[0] == time and levels[-1] == 1
        assert all(a <= b for a, b in zip(levels[:-1], levels[1:]))
    assert alpha(torch.tensor(.8)) < alpha(torch.tensor(.2))


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("budget", [1, 2, 8])
def test_update_has_exact_budget_and_does_not_mutate_state(action, budget):
    model = ZeroDenoiser()
    sampler = Sampler(model)
    x = torch.randn(4, 3, 2)
    original = x.clone()
    samples = sampler.update(State(x, .3, 7), torch.zeros(1, 8), torch.ones(1, 8),
                             action, budget, torch.randn_like(x))
    assert model.calls == budget
    assert samples.shape == x.shape and torch.isfinite(samples).all()
    assert torch.equal(x, original)


def test_restart_is_independent_of_interrupted_state():
    sampler = Sampler(ZeroDenoiser())
    noise, context = torch.randn(4, 3, 2), torch.zeros(1, 8)
    one = sampler.update(State(torch.zeros_like(noise), .3, 4), context, context, "restart", 3, noise)
    two = sampler.update(State(torch.ones_like(noise) * 100, .9, 9), context, context, "restart", 3, noise)
    torch.testing.assert_close(one, two)


def test_forward_transition_matches_conditional_vp():
    class Capture(Sampler):
        def denoise(self, x, start, steps, context, old_context=None, end=0):
            self.start = start
            return x
    sampler = Capture(None)
    x, noise = torch.randn(3, 2, 2), torch.randn(3, 2, 2)
    context = torch.zeros(1, 8)
    out = sampler.update(State(x, .3, 4), context, context, "fixed50", 2, noise)
    ratio = alpha(torch.tensor(sampler.start)) / alpha(torch.tensor(.3))
    torch.testing.assert_close(out, ratio.sqrt() * x + (1 - ratio).sqrt() * noise)


def test_masks_prevent_hidden_observation_leakage():
    model = TrajectoryDiffusion(obs_len=4, pred_len=3, hidden=16, layers=1).eval()
    p = torch.randn(2, 3, 4, 2)
    mask = torch.rand(2, 3, 4) > .5
    mask[:, 0, -1] = True
    changed = p.clone()
    changed[~mask] = 10000
    torch.testing.assert_close(model.encoder(p, mask), model.encoder(changed, mask))
    # An empty padded neighbor contributes nothing, irrespective of coordinates.
    extra = torch.randn(2, 1, 4, 2)
    torch.testing.assert_close(model.encoder(p, mask), model.encoder(
        torch.cat((p, extra), 1), torch.cat((mask, torch.zeros(2, 1, 4, dtype=torch.bool)), 1)))


def test_model_loss_backward_and_checkpoint(tmp_path):
    torch.manual_seed(1)
    model = TrajectoryDiffusion(obs_len=4, pred_len=3, hidden=16, layers=1)
    old = torch.ones(2, 3, 4, dtype=torch.bool)
    old[:, 1:, 2:] = False
    batch = dict(positions=torch.randn(2, 3, 4, 2), old_mask=old,
                 new_mask=torch.ones_like(old), future=torch.randn(2, 3, 2))
    loss = diffusion_loss(model, batch, .1)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    path = tmp_path / "model.pt"
    save_checkpoint(path, dict(model=model.state_dict(), model_config=model.config))
    loaded = load_model(path, "cpu")
    for a, b in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(a, b)


def test_selector_features_and_scores():
    old, new = torch.randn(1, 16), torch.randn(1, 16)
    mask = torch.ones(1, 3, 4, dtype=torch.bool)
    x = features(old, new, mask, mask, .3, 4)
    assert x.shape == (4 * 16 + 8,)
    selector = DepthSelector(len(x))
    assert selector(x).shape == (5,)
    assert selector.choose(x) in ACTIONS
    assert not torch.equal(x, features(new, old, mask, mask, .3, 4))


def test_energy_score_known_two_point_distribution():
    samples = torch.tensor([[[0., 0.]], [[2., 0.]]])
    truth = torch.tensor([[0., 0.]])
    assert float(energy_score(samples, truth)) == pytest.approx(0)
    # U-statistic can be negative for a finite ensemble; do not clamp it.
    assert float(energy_score(samples, torch.tensor([[1., 0.]]))) == pytest.approx(0)
    summary = metrics(samples, truth)
    assert summary["min_ade"] == 0 and summary["diversity"] == 2
    with pytest.raises(ValueError):
        energy_score(samples[:1], truth)


def record(scene):
    return dict(positions=np.zeros((1, 2, 4, 2), np.float32),
                old_mask=np.ones((1, 2, 4), bool), new_mask=np.ones((1, 2, 4), bool),
                future=np.zeros((1, 3, 2), np.float32), scene=np.array([scene]))


def test_scene_leakage_is_rejected(tmp_path):
    for split in ("train", "val", "test"):
        np.savez(tmp_path / f"{split}.npz", **record("same-scene"))
    with pytest.raises(ValueError, match="leakage"):
        load_splits(tmp_path)


def test_invalid_disclosure_mask_is_rejected(tmp_path):
    data = record("one")
    data["new_mask"][0, 1, 0] = False
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="subset"):
        Trajectories(path)


def test_eth_preparation_uses_only_historical_context(tmp_path):
    specs = []
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.txt"
        rows = [[f, ped, f + ped, ped] for f in range(8) for ped in (0, 1)]
        np.savetxt(path, rows)
        specs.append(dict(file=path.name, scene=split, split=split))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(sequences=specs)))
    out = tmp_path / "prepared"
    prepare_eth(tmp_path, manifest, out, obs_len=4, pred_len=3, agents=3, frame_step=1, stride=1)
    splits = load_splits(out)
    item = splits["train"][0]
    assert item["positions"].shape == (3, 4, 2)
    assert item["future"][0, 0] == 1
    assert item["positions"][0, -1, 0] == 0
    assert not item["new_mask"][2].any()


def test_calibration_uses_validation_action_scores(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 4 * 8 + 8)).astype(np.float32)
    for split in ("train", "val"):
        scores = np.ones((20, 5), np.float32)
        scores[:, 3] = 0
        np.savez(tmp_path / f"{split}.npz", features=x, scores=scores)
    (tmp_path / "metadata.json").write_text(json.dumps(dict(model_sha256="abc", experiment={})))
    calibrate(tmp_path, tmp_path / "calibration.json")
    result = json.loads((tmp_path / "calibration.json").read_text())
    assert result["fixed_action"] == "fixed75"
    assert len(result["l2"]) == 4


def test_default_config_is_valid():
    config = read_config("configs/default.json")
    assert config["experiment"]["samples"] >= 2


@pytest.mark.parametrize("budget", [2, 4])
def test_comparison_pipeline_charges_diagnostic_calls(tmp_path, monkeypatch, budget):
    # Inject one artificial state and a zero denoiser. No dataset or benchmark run.
    model = ZeroDenoiser()
    checkpoint = tmp_path / "diffusion.pt"
    checkpoint.write_bytes(b"unit-test-checkpoint")
    config = read_config("configs/default.json")
    cfg = config["experiment"]
    cfg["budgets"], cfg["samples"] = [budget], 4
    old_h, new_h = torch.zeros(1, 8), torch.ones(1, 8)
    mask = torch.ones(1, 2, 4, dtype=torch.bool)
    x = features(old_h, new_h, mask, mask, .5, budget)
    task = dict(index=0, scene="fixture", seed=1, fraction=.5, budget=budget,
                state=State(torch.randn(4, 3, 2), .5, 8), noise=torch.randn(4, 3, 2),
                old_h=old_h, new_h=new_h, truth=torch.zeros(3, 2), features=x)
    selector = DepthSelector(len(x))
    selector_path = tmp_path / "selector.pt"
    save_checkpoint(selector_path, dict(selector=selector.state_dict(), feature_dim=len(x),
        model_sha256=file_hash(checkpoint), experiment=cfg))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(dict(model_sha256=file_hash(checkpoint),
        fixed_action="fixed50", l2=[.1, .2, .3, .4], cosine=[.1, .2, .3, .4])))
    (tmp_path / "test.npz").write_bytes(b"unit-test-data")
    monkeypatch.setattr(experiment, "load_model", lambda *args: model)
    monkeypatch.setattr(experiment, "load_splits", lambda *args: {"test": None})
    monkeypatch.setattr(experiment, "trials", lambda *args: iter([task]))
    output = tmp_path / "comparison.csv"
    experiment.evaluate(config, tmp_path, checkpoint, selector_path, calibration, output)
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(cfg["methods"])
    assert model.calls == budget * len(cfg["methods"])
    assert all(int(r["update_nfe"]) == budget for r in rows)
    probe = next(r for r in rows if r["method"] == "probe")
    assert int(probe["probe_nfe"]) == (2 if budget > 2 else 0)
    assert all(int(r["total_nfe"]) == budget + 8 for r in rows)
    summary = tmp_path / "summary.json"
    experiment.summarize(output, summary)
    assert len(json.loads(summary.read_text())) == len(cfg["methods"])
