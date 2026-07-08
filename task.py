"""
task.py
-------
Two tasks:
  - Pattern generation (debug/reservoir): Frozen inputs -> sum of sines (MSE).
  - Evidence accumulation (HEADLINE, Bellec 2020 Fig.3): Random cue sequence for each trial.
    The network must accumulate left/right cues in memory and select the majority side in the decision window.
    Recurrent plasticity is REQUIRED (no memorization possible; requires working memory + counting).
"""
from __future__ import annotations
import math
import torch


# ============================ PATTERN GENERATION ============================
def make_pattern_task(tcfg, torch_device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(tcfg.seed)
    T, n_in, n_out = tcfg.T, tcfg.n_in, tcfg.n_out
    X = (torch.rand(T, n_in, generator=g) < tcfg.input_rate).to(dtype)
    t = torch.arange(T, dtype=dtype).unsqueeze(1)
    Y = torch.zeros(T, n_out, dtype=dtype)
    for f, a in zip(tcfg.freqs, tcfg.amps):
        Y[:, 0] += a * torch.sin(2 * math.pi * f * t[:, 0] / T)
    if len(tcfg.amps):
        Y = Y / (sum(abs(a) for a in tcfg.amps))
    return X.to(torch_device), Y.to(torch_device)


# ========================== EVIDENCE ACCUMULATION ==========================
def evidence_geometry(tc):
    """Derives n_in and T for the evidence task based on cue parameters."""
    n_in = 4 * tc.n_group
    T = tc.n_cues * (tc.cue_dur + tc.gap) + tc.delay + tc.decision
    return n_in, T


def make_evidence_trial(tc, gen, torch_device="cpu", dtype=torch.float32):
    """
    Generates a single trial of the evidence accumulation task.
    Returns: X [T, n_in], Ystar [T, 2] (one-hot, active in decision window), mask [T], label (0=left, 1=right)
    Input channel groups:
      - [0 : n_group]: Left cue channels
      - [n_group : 2*n_group]: Right cue channels
      - [2*n_group : 3*n_group]: Recall cue channels
      - [3*n_group : 4*n_group]: Background noise channels
    """
    ng = tc.n_group
    n_in, T = evidence_geometry(tc)

    # Background noise (low-rate spikes across all channels)
    X = (torch.rand(T, n_in, generator=gen) < tc.noise_rate).to(dtype)

    # Cues: each cue randomly selected as left or right
    sides_left = (torch.rand(tc.n_cues, generator=gen) < tc.p_left)   # True = Left
    for c in range(tc.n_cues):
        start = c * (tc.cue_dur + tc.gap)
        grp = slice(0, ng) if bool(sides_left[c]) else slice(ng, 2 * ng)
        spikes = (torch.rand(tc.cue_dur, ng, generator=gen) < tc.cue_rate).to(dtype)
        X[start:start + tc.cue_dur, grp] = spikes

    # Recall cue (tells SNN to output decision during the decision window)
    dec_start = tc.n_cues * (tc.cue_dur + tc.gap) + tc.delay
    recall = (torch.rand(tc.decision, ng, generator=gen) < tc.cue_rate).to(dtype)
    X[dec_start:dec_start + tc.decision, 2 * ng:3 * ng] = recall

    # Label: majority side (n_cues is odd -> no ties)
    n_left = int(sides_left.sum().item())
    label = 0 if n_left > (tc.n_cues - n_left) else 1        # 0 = Left, 1 = Right

    Ystar = torch.zeros(T, 2, dtype=dtype)
    mask = torch.zeros(T, dtype=dtype)
    Ystar[dec_start:dec_start + tc.decision, label] = 1.0
    mask[dec_start:dec_start + tc.decision] = 1.0

    return (X.to(torch_device), Ystar.to(torch_device),
            mask.to(torch_device), label)