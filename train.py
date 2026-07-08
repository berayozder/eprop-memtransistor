"""
train.py
--------
Configures the SNN experiment + executes the episodic e-prop training loop.
Supports two tasks:
  - pattern: Static X, Y pairs (regression/MSE loss)
  - evidence: Dynamically generated cue sequence per trial (classification/cross-entropy loss, tracking accuracy)
Each trial: runs network forward pass -> obtains gradients -> updates (W_rec, W_in) on device via Synapse/Writer,
while readout parameters are updated ideally (or on-device optionally / can be frozen).
"""
from __future__ import annotations
import math
import torch

from device_interface import IdealDevice
from memtransistor import Memtransistor
from synapse import Synapse
from network import LSNN
from task import make_pattern_task, make_evidence_trial, evidence_geometry


def _make_device(shape, dcfg, torch_device, seed):
    if dcfg.kind == "ideal":
        return IdealDevice(shape, torch_device=torch_device, seed=seed)
    elif dcfg.kind == "memtransistor":
        return Memtransistor(shape, dcfg, torch_device=torch_device, seed=seed)
    raise ValueError(dcfg.kind)


def build(cfg):
    nc, tc, dc, sc, trc = cfg.neuron, cfg.task, cfg.device, cfg.synapse, cfg.train
    dev = trc.torch_device
    g = torch.Generator(device="cpu").manual_seed(trc.seed)
    n = nc.n_rec

    # --- Task geometry setup (derive n_in, T, n_out for the evidence accumulation task) ---
    if tc.kind == "evidence":
        n_in, _T = evidence_geometry(tc)
        tc.n_in, tc.n_out = n_in, 2
        n_out = 2
    else:
        n_in, n_out = tc.n_in, tc.n_out

    # --- Initial weights setup ---
    W_rec0 = (trc.w_gain / math.sqrt(n)) * torch.randn(n, n, generator=g)
    W_rec0.fill_diagonal_(0.0)
    W_in0 = (trc.w_gain / math.sqrt(n_in)) * torch.randn(n, n_in, generator=g)
    W_out0 = (1.0 / math.sqrt(n)) * torch.randn(n_out, n, generator=g)
    b_out = torch.zeros(n_out, device=dev)

    # --- Synapse and Device setup (W_rec and W_in are on-device) ---
    syn_rec = Synapse(_make_device((n, n), dc, dev, trc.seed + 1),
                      sc.w_range, sc.writer, sc.verify_max_iter)
    syn_in = Synapse(_make_device((n, n_in), dc, dev, trc.seed + 2),
                     sc.w_range, sc.writer, sc.verify_max_iter)
    syn_rec.init_weight(W_rec0.to(dev))
    syn_in.init_weight(W_in0.to(dev))

    # --- Readout setup: ideal (default) or on-device (ablation/noise propagation) ---
    if trc.readout_on_device:
        syn_out = Synapse(_make_device((n_out, n), dc, dev, trc.seed + 3),
                          sc.w_range, sc.writer, sc.verify_max_iter)
        syn_out.init_weight(W_out0.to(dev))
        readout = syn_out
    else:
        readout = W_out0.to(dev)

    # --- Learning-signal feedback (symmetric tracks W_out, random uses static B_fixed) ---
    if trc.eprop_variant == "symmetric":
        B_fixed = None
    elif trc.eprop_variant == "random":
        B_fixed = ((1.0 / math.sqrt(n)) * torch.randn(n_out, n, generator=g)).to(dev)
    else:
        raise ValueError(trc.eprop_variant)

    net = LSNN(nc, tc, syn_rec, syn_in, readout, b_out,
               variant=trc.eprop_variant, B_fixed=B_fixed, torch_device=dev)

    # --- Setup task data ---
    if tc.kind == "evidence":
        task = {"kind": "evidence", "tc": tc,
                "gen": torch.Generator(device="cpu").manual_seed(tc.seed)}
    else:
        X, Y = make_pattern_task(tc, torch_device=dev)
        task = {"kind": "pattern", "X": X, "Y": Y}
    return net, task


def _apply_update(net, res, lr, readout_trainable):
    net.syn_rec.update(-lr * res["grad_rec"])
    net.syn_in.update(-lr * res["grad_in"])
    if readout_trainable:
        if net.readout_is_device:
            net.readout.update(-lr * res["grad_out"])
        else:
            net.readout.add_(-lr * res["grad_out"])
        net.b_out.add_(-lr * res["grad_b"])


def _pulses(net):
    p = net.syn_rec.n_pulses_total + net.syn_in.n_pulses_total
    if net.readout_is_device:
        p += net.readout.n_pulses_total
    return p


def train(cfg, verbose=True):
    trc, tc = cfg.train, cfg.task
    dev = trc.torch_device
    net, task = build(cfg)
    lr = trc.lr
    history = {"trial": [], "loss": [], "acc": [], "pulses": []}
    acc_ma = None                              # accuracy moving average (only for evidence task)

    for it in range(trc.n_trials):
        if task["kind"] == "evidence":
            X, Ystar, mask, label = make_evidence_trial(task["tc"], task["gen"], torch_device=dev)
            res = net.run_trial(X, Ystar, mask=mask, loss="classification")
            dec = int((res["y"] * mask[:, None]).sum(0).argmax().item())
            correct = float(dec == label)
            acc_ma = correct if acc_ma is None else 0.98 * acc_ma + 0.02 * correct
        else:
            res = net.run_trial(task["X"], task["Y"], loss="regression")

        _apply_update(net, res, lr, trc.readout_trainable)

        if it % trc.log_every == 0 or it == trc.n_trials - 1:
            history["trial"].append(it)
            history["loss"].append(res["loss"])
            history["acc"].append(acc_ma if acc_ma is not None else float("nan"))
            history["pulses"].append(_pulses(net))
            if verbose:
                acc_s = f" | acc {acc_ma:.3f}" if acc_ma is not None else ""
                print(f"  trial {it:5d} | loss {res['loss']:.4f}{acc_s} | pulses {history['pulses'][-1]}")

    return net, history