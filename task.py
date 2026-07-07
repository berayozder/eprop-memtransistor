"""
task.py
-------
Pattern generation task: maps a frozen (static) input spike raster to a target wave form (sum of sines).
This represents the classic e-prop sanity check task (Bellec et al. 2020, Supplementary Figure 1).
The input remains static across trials, while training progresses across trials.
"""
from __future__ import annotations
import math
import torch


def make_pattern_task(tcfg, torch_device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(tcfg.seed)
    T, n_in, n_out = tcfg.T, tcfg.n_in, tcfg.n_out

    # Frozen input: Bernoulli spike raster
    X = (torch.rand(T, n_in, generator=g) < tcfg.input_rate).to(dtype)

    # Target: 0-centered sum of sines, normalized close to [-1, 1] range
    t = torch.arange(T, dtype=dtype).unsqueeze(1)          # [T, 1]
    Y = torch.zeros(T, n_out, dtype=dtype)
    for f, a in zip(tcfg.freqs, tcfg.amps):
        Y[:, 0] += a * torch.sin(2 * math.pi * f * t[:, 0] / T)
    if len(tcfg.amps):
        Y = Y / (sum(abs(a) for a in tcfg.amps))           # Normalize
     
    return X.to(torch_device), Y.to(torch_device)