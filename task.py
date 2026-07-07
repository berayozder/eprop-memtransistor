"""
task.py
-------
Pattern generation: sabit (frozen) girdi spike raster'i -> hedef dalga formu
(sinus toplami). e-prop'un klasik sanity gorevi (Bellec Supplementary Fig.1).
Girdi trial'lar boyunca SABIT; ogrenme iterasyonlar boyunca gerceklesir.
"""
from __future__ import annotations
import math
import torch


def make_pattern_task(tcfg, torch_device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(tcfg.seed)
    T, n_in, n_out = tcfg.T, tcfg.n_in, tcfg.n_out

    # frozen girdi: Bernoulli spike raster
    X = (torch.rand(T, n_in, generator=g) < tcfg.input_rate).to(dtype)

    # hedef: sinus toplami (0-merkezli), [-1,1] civari
    t = torch.arange(T, dtype=dtype).unsqueeze(1)          # [T,1]
    Y = torch.zeros(T, n_out, dtype=dtype)
    for f, a in zip(tcfg.freqs, tcfg.amps):
        Y[:, 0] += a * torch.sin(2 * math.pi * f * t[:, 0] / T)
    if len(tcfg.amps):
        Y = Y / (sum(abs(a) for a in tcfg.amps))           # normalize

    return X.to(torch_device), Y.to(torch_device)