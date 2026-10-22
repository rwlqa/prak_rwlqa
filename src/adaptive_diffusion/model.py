"""A compact, mask-aware social encoder and conditional epsilon denoiser.

This is an independent implementation, not a reproduction/checkpoint of MID.
Positions are in metres relative to the target's last observed position.
"""
import math

import torch
from torch import nn


class ContextEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.history = nn.GRU(4, hidden, batch_first=True)
        self.social = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.SiLU())
        self.combine = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())

    def forward(self, positions, mask):
        # [batch, agents, observations, xy]. A masked value cannot affect h.
        b, a, o, _ = positions.shape
        age = torch.linspace(0, 1, o, device=positions.device).view(1, 1, o, 1)
        features = torch.cat((positions * mask[..., None], mask[..., None].float(),
                              age.expand(b, a, o, 1) * mask[..., None]), dim=-1)
        _, h = self.history(features.reshape(b * a, o, 4))
        h = h[-1].reshape(b, a, -1)
        valid = mask.any(-1)
        # Last available position, rather than an unobserved last slot.
        ids = torch.arange(o, device=positions.device).view(1, 1, o)
        last = (ids * mask).amax(-1)
        xy = positions.gather(2, last[..., None, None].expand(b, a, 1, 2)).squeeze(2)
        social = self.social(torch.cat((h, xy), -1))
        valid = valid.clone()
        valid[:, 0] = False
        pooled = (social * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.combine(torch.cat((h[:, 0], pooled), -1))


class TrajectoryDiffusion(nn.Module):
    def __init__(self, obs_len=8, pred_len=12, hidden=64, layers=2):
        super().__init__()
        if hidden % 4 or hidden < 8 or min(obs_len, pred_len, layers) < 1:
            raise ValueError("hidden must be >=8 and divisible by 4; lengths/layers positive")
        self.config = dict(obs_len=obs_len, pred_len=pred_len, hidden=hidden, layers=layers)
        self.encoder = ContextEncoder(hidden)
        self.input = nn.Linear(2, hidden)
        self.position = nn.Parameter(torch.randn(pred_len, hidden) * 0.01)
        self.condition = nn.Sequential(nn.Linear(3 * hidden + 16, hidden), nn.SiLU(),
                                       nn.Linear(hidden, hidden))
        block = nn.TransformerEncoderLayer(hidden, 4, 4 * hidden, dropout=0,
                                            activation="gelu", batch_first=True)
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.output = nn.Linear(hidden, 2)

    def forward(self, x, t, context, old_context=None):
        if old_context is None:
            old_context = context
        freq = torch.exp(torch.linspace(0, math.log(1000), 8, device=x.device))
        angle = t[:, None] * freq[None]
        time = torch.cat((angle.sin(), angle.cos()), -1)
        condition = self.condition(torch.cat((context, old_context,
                                              context - old_context, time), -1))
        tokens = self.input(x) + self.position + condition[:, None]
        return self.output(self.blocks(tokens))
