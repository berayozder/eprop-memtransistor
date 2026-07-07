"""
train.py
--------
Configures the SNN experiment + executes the episodic e-prop training loop.
Each trial: runs network forward pass -> obtains gradients -> updates (W_rec, W_in) on device via Synapse/Writer,
while readout parameters (W_out, b_out) are updated ideally (or on-device optionally).
"""
from __future__ import annotations
import math
import torch

from device_interface import IdealDevice
from memtransistor import Memtransistor
from synapse import Synapse
from network import LSNN
from task import make_pattern_task


def _make_device(shape, dcfg, torch_device, seed):
    if dcfg.kind == "ideal":
        return IdealDevice(shape, torch_device=torch_device, seed=seed)
    elif dcfg.kind == "memtransistor":
        return Memtransistor(shape, dcfg, torch_device=torch_device, seed=seed)
    raise ValueError(dcfg.kind)


def build(cfg):
    nc, tc, dc, sc, trc = cfg.neuron, cfg.task, cfg.device, cfg.synapse, cfg.train
    dev, dt = trc.torch_device, torch.float32
    g = torch.Generator(device="cpu").manual_seed(trc.seed)
    n, n_in, n_out = nc.n_rec, tc.n_in, tc.n_out

    # --- Initial weights setup ---
    W_rec0 = (trc.w_gain / math.sqrt(n)) * torch.randn(n, n, generator=g)
    W_rec0.fill_diagonal_(0.0)
    W_in0 = (trc.w_gain / math.sqrt(n_in)) * torch.randn(n, n_in, generator=g)
    W_out = (1.0 / math.sqrt(n)) * torch.randn(n_out, n, generator=g)
    W_out = W_out.to(dev)
    b_out = torch.zeros(n_out, device=dev)

    # --- Synapse and Device setup (W_rec and W_in are on-device) ---
    syn_rec = Synapse(_make_device((n, n), dc, dev, trc.seed + 1),
                      sc.w_range, sc.writer, sc.verify_max_iter)
    syn_in = Synapse(_make_device((n, n_in), dc, dev, trc.seed + 2),
                     sc.w_range, sc.writer, sc.verify_max_iter)
    syn_rec.init_weight(W_rec0.to(dev))
    syn_in.init_weight(W_in0.to(dev))

    # --- Learning-signal feedback B ---
    if trc.eprop_variant == "symmetric":
        B_rec = W_out                                  # Tracks W_out (shares the same tensor)
    elif trc.eprop_variant == "random":
        B_rec = (1.0 / math.sqrt(n)) * torch.randn(n_out, n, generator=g)
        B_rec = B_rec.to(dev)                           # Static random feedback
    else:
        raise ValueError(trc.eprop_variant)

    net = LSNN(nc, tc, syn_rec, syn_in, W_out, b_out, B_rec, torch_device=dev)
    X, Y = make_pattern_task(tc, torch_device=dev)
    return net, X, Y


def train(cfg, verbose=True):
    trc = cfg.train
    net, X, Y = build(cfg)
    lr = trc.lr
    history = {"trial": [], "loss": [], "pulses": []}

    for it in range(trc.n_trials):
        res = net.run_trial(X, Y)

        # Write weights updates: W_rec, W_in -> writer -> device
        net.syn_rec.update(-lr * res["grad_rec"])
        net.syn_in.update(-lr * res["grad_in"])

        # Readout updates (ideal updates -> symmetric B_rec automatically follows)
        net.W_out.add_(-lr * res["grad_out"])
        net.b_out.add_(-lr * res["grad_b"])

        if it % trc.log_every == 0 or it == trc.n_trials - 1:
            pulses = net.syn_rec.n_pulses_total + net.syn_in.n_pulses_total
            history["trial"].append(it)
            history["loss"].append(res["loss"])
            history["pulses"].append(pulses)
            if verbose:
                print(f"  trial {it:5d} | loss {res['loss']:.5f} | pulses {pulses}")

    return net, history