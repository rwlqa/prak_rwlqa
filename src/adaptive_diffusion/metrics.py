"""Finite-ensemble proper scores, displacement, and descriptive spread."""
import torch


def energy_score(samples, truth):
    # U-statistic (off diagonal): unbiased estimator of E||X-y||-.5 E||X-X'||.
    flat = samples.flatten(1)
    k = len(flat)
    if k < 2:
        raise ValueError("Energy Score requires >=2 independent samples")
    first = torch.linalg.vector_norm(flat - truth.flatten(), dim=-1).mean()
    second = torch.cdist(flat, flat).sum() / (k * (k - 1))
    return first - .5 * second


def metrics(samples, truth):
    distances = torch.linalg.vector_norm(samples - truth, dim=-1)
    k, h, _ = samples.shape
    flat, y = samples.flatten(1), truth.flatten()
    i, j = torch.triu_indices(len(y), len(y), offset=1, device=y.device)
    expected = (flat[:, i] - flat[:, j]).abs().sqrt().mean(0)
    observed = (y[i] - y[j]).abs().sqrt()
    # Marginal intervals are descriptive, not simultaneous trajectory coverage.
    low, high = torch.quantile(samples, samples.new_tensor([.05, .95]), dim=0)
    return {
        "energy_score": float(energy_score(samples, truth)),
        "variogram_score": float((expected - observed).square().mean()),
        "min_ade": float(distances.mean(-1).min()),
        "min_fde": float(distances[:, -1].min()),
        "mean_ade": float(distances.mean()),
        "marginal_coverage90": float(((truth >= low) & (truth <= high)).float().mean()),
        "interval_width90": float((high - low).mean()),
        "diversity": float(torch.cdist(flat, flat).sum() / (k * (k - 1))),
    }
