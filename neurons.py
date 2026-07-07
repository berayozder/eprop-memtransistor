"""
neurons.py
----------
LIF/ALIF primitifleri. Asil ileri-gecis + eligibility recursion network.py'de
(trace'ler noron dinamigine sikica bagli oldugu icin tek yerde tutmak daha az hatali).
"""
from __future__ import annotations
import torch


def heaviside(x):
    return (x > 0).to(x.dtype)


def pseudo_derivative(v, A, v_th, gamma_pd):
    """
    psi = (gamma/v_th) * max(0, 1 - |v - A|/v_th)
    LIF icin A = v_th. (Bellec 2020, Methods: pseudo derivative)
    """
    return (gamma_pd / v_th) * torch.clamp(1.0 - torch.abs(v - A) / v_th, min=0.0)