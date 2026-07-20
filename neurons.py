"""
neurons.py
==========
LOW-LEVEL NEURON PRIMITIVES: the spike function and its surrogate gradient.

This file is intentionally tiny. The actual neuron dynamics (membrane update,
recurrence) and the eligibility-trace recursions live in network.py, because those
traces are tightly coupled to the forward pass and keeping them in one place makes
bugs far less likely. Here we only define the two building blocks that both the
forward pass and e-prop need.

Background: a spiking neuron fires (outputs 1) when its membrane voltage crosses a
threshold, else outputs 0. That step function has a zero/undefined derivative, which
would make gradient-based learning impossible. e-prop (like most surrogate-gradient
methods) replaces the derivative of the step with a smooth "pseudo-derivative" -- a
triangular bump centred at the threshold. That is what `pseudo_derivative` computes.
"""
from __future__ import annotations
import torch


def heaviside(x):
    """Hard spike function: returns 1.0 where x > 0 (neuron fires), else 0.0.

    Args:
        x: membrane-minus-threshold value (v - A). Positive means "above threshold".
    Returns:
        A 0/1 tensor of the same shape as x.
    """
    return (x > 0).to(x.dtype)


def pseudo_derivative(v, A, v_th, gamma_pd):
    """Surrogate gradient of the spike function (Bellec 2020, Methods).

        psi = (gamma_pd / v_th) * max(0, 1 - |v - A| / v_th)

    This is a triangular bump: it is largest (= gamma_pd / v_th) exactly at the
    threshold (v == A) and falls linearly to 0 once the membrane is more than v_th
    away from the threshold in either direction. It tells the learning rule "how
    sensitive was this neuron's firing to a small change in its input right now".

    Args:
        v:        membrane voltage (per neuron).
        A:        effective threshold (v_th for LIF; v_th + beta*a for ALIF).
        v_th:     base threshold, also sets the width of the bump.
        gamma_pd: overall amplitude (dampening) of the surrogate gradient.
    Returns:
        psi: the pseudo-derivative, same shape as v.
    """
    return (gamma_pd / v_th) * torch.clamp(1.0 - torch.abs(v - A) / v_th, min=0.0)