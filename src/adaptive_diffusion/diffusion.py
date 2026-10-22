"""Continuous cosine VP schedule, DDIM (eta=0), and exact forward transitions."""
from dataclasses import dataclass
import math

import torch

ACTIONS = ("continue", "fixed25", "fixed50", "fixed75", "restart")


def alpha(t):
    return torch.cos((t + 0.008) / 1.008 * math.pi / 2).square().clamp(1e-5, 1 - 1e-5)


def logsnr(t):
    a = alpha(t)
    return torch.log(a) - torch.log1p(-a)


def correction_time(t: float, action: str) -> float:
    if action not in ACTIONS:
        raise ValueError(action)
    if action == "continue":
        return t
    if action == "restart":
        return 1.0
    fraction = {"fixed25": .25, "fixed50": .5, "fixed75": .75}[action]
    start, end = logsnr(torch.tensor(t)), logsnr(torch.tensor(1.0))
    target = start + fraction * (end - start)
    a = torch.sigmoid(target)
    return float((2 / math.pi * torch.acos(a.sqrt()) * 1.008 - .008).clamp(t, 1))


def randn(shape, device, generator):
    # MPS has no native torch.Generator; a CPU stream can still be replayed.
    return torch.randn(shape, device=generator.device, generator=generator).to(device)


@dataclass
class State:
    x: torch.Tensor  # [ensemble, horizon, 2], a single scene
    time: float
    prefix_nfe: int


class Sampler:
    def __init__(self, model):
        self.model = model

    def step(self, x, t: float, next_t: float, context, old_context=None):
        ts = x.new_full((len(x),), t)
        eps = self.model(x, ts, context.expand(len(x), -1),
                         None if old_context is None else old_context.expand(len(x), -1))
        a = alpha(ts)[:, None, None]
        clean = (x - (1 - a).sqrt() * eps) / a.sqrt()
        # The clean endpoint is exact; schedule alpha(0) itself is slightly <1.
        if next_t == 0:
            return clean
        next_a = alpha(x.new_tensor(next_t))
        return next_a.sqrt() * clean + (1 - next_a).sqrt() * eps

    def denoise(self, x, start, steps, context, old_context=None, end=0.0):
        if steps < 1 or not 0 <= end < start <= 1:
            raise ValueError("Need positive NFE and 0 <= end < start <= 1")
        grid = torch.linspace(start, end, steps + 1).tolist()
        for t, nxt in zip(grid[:-1], grid[1:]):
            x = self.step(x, t, nxt, context, old_context)
        return x

    @torch.no_grad()
    def interrupt(self, context, samples, pred_len, fraction, prefix_steps, generator):
        if not 0 < fraction < 1 or prefix_steps < 1 or samples < 2:
            raise ValueError("Invalid interruption or ensemble size")
        # Arrival is a fraction of nominal NFE. Round explicitly and log actual time.
        steps = max(1, min(prefix_steps - 1, round(prefix_steps * fraction)))
        if prefix_steps < 2:
            raise ValueError("prefix_steps must be >=2")
        time = 1 - steps / prefix_steps
        x = randn((samples, pred_len, 2), context.device, generator)
        x = self.denoise(x, 1, steps, context, end=time)
        return State(x, time, steps)

    @torch.no_grad()
    def update(self, state, old_context, new_context, action, budget, noise):
        time = correction_time(state.time, action)
        if noise.shape != state.x.shape:
            raise ValueError("Noise must match interrupted ensemble")
        if action == "restart":
            x = noise.clone()  # Independent of the old ensemble.
            previous = new_context
        else:
            ratio = alpha(state.x.new_tensor(time)) / alpha(state.x.new_tensor(state.time))
            ratio = ratio.clamp(0, 1)
            x = ratio.sqrt() * state.x + (1 - ratio).sqrt() * noise
            previous = old_context
        return self.denoise(x, time, budget, new_context, previous)
