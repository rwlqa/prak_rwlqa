"""Predict action-wise expected Energy Score from two adjacent context states."""
import torch
from torch import nn
from torch.nn import functional as F

from .diffusion import ACTIONS, logsnr


def features(old_h, new_h, old_mask, new_mask, time, budget):
    delta = new_h - old_h
    age = torch.linspace(0, 1, old_mask.shape[-1], device=old_h.device)
    scalar = old_h.new_tensor([
        float(logsnr(old_h.new_tensor(time))) / 10,
        budget / 64,
        float(old_mask.float().mean()), float(new_mask.float().mean()),
        float((new_mask & ~old_mask).float().mean()),
        float((old_mask * age).sum() / old_mask.sum().clamp_min(1)),
        float((new_mask * age).sum() / new_mask.sum().clamp_min(1)),
        float(F.cosine_similarity(old_h, new_h).item()),
    ])[None]
    return torch.cat((old_h, new_h, delta, delta.abs(), scalar), -1).squeeze(0)


class DepthSelector(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, 128), nn.SiLU(),
                                     nn.Linear(128, 64), nn.SiLU(),
                                     nn.Linear(64, len(ACTIONS)))
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("scale", torch.ones(feature_dim))
        self.register_buffer("score_scale", torch.tensor(1.0))

    def forward(self, x):
        return self.network((x - self.mean) / self.scale) * self.score_scale

    @torch.no_grad()
    def choose(self, x):
        return ACTIONS[int(self(x).argmin(-1))]


def distance_action(old_h, new_h, kind, thresholds):
    if kind == "l2":
        distance = float((new_h - old_h).square().mean().sqrt())
    elif kind == "cosine":
        distance = float(1 - F.cosine_similarity(old_h, new_h).item())
    else:
        raise ValueError(kind)
    return ACTIONS[sum(distance > threshold for threshold in thresholds)]
