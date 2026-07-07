"""
neurons.py
----------
LIF/ALIF neural primitives.
The core temporal forward pass and eligibility recurrence loop are implemented in network.py
(grouping state variables and traces in a single place reduces programming bugs).
"""
from __future__ import annotations
import torch


def heaviside(x):
    return (x > 0).to(x.dtype)


def pseudo_derivative(v, A, v_th, gamma_pd):
    """
    Computes the pseudo-derivative psi for e-prop gradient calculations:
    psi = (gamma_pd / v_th) * max(0, 1 - |v - A| / v_th)
    For standard LIF neurons, A = v_th (Bellec 2020, Methods: pseudo derivative).
    """
    return (gamma_pd / v_th) * torch.clamp(1.0 - torch.abs(v - A) / v_th, min=0.0)