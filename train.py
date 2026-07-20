"""
train.py
========
ASSEMBLES THE SYSTEM and RUNS THE TRAINING LOOP.

build(cfg) wires everything together: it creates the neurons, the device(s), the
synapses (mapping weights to conductances), the readout, and the task. train(cfg)
then runs the episodic e-prop loop: for each trial, run the network forward, get the
desired weight changes from e-prop, and write them onto the device via the synapse
writer. The recurrent and input weights live on the device; the readout is ideal by
default (or on device / frozen for ablations).

Reading order: this file ties together config, memtransistor, synapse, network, task.
"""
from __future__ import annotations
import math
import torch

from device_interface import IdealDevice
from memtransistor import Memtransistor
from synapse import Synapse
from network import LSNN
from task import make_evidence_trial, evidence_geometry


def _make_device(shape, dcfg, torch_device, seed):
    """Create one device array of the given shape: perfect (ideal) or realistic (memtransistor)."""
    if dcfg.kind == "ideal":
        return IdealDevice(shape, torch_device=torch_device, seed=seed)
    elif dcfg.kind == "memtransistor":
        return Memtransistor(shape, dcfg, torch_device=torch_device, seed=seed)
    raise ValueError(dcfg.kind)


def build(cfg):
    """Construct the full network + task from a config.

    Returns:
        net:  the LSNN, with its recurrent/input weights placed on device(s).
        task: a dict describing the evidence accumulation generator.
              generator that produces a fresh random trial each time).
    """
    nc, tc, dc, sc, trc = cfg.neuron, cfg.task, cfg.device, cfg.synapse, cfg.train
    dev = trc.torch_device
    g = torch.Generator(device="cpu").manual_seed(trc.seed)   # RNG for weight init
    n = nc.n_rec

    # --- Task geometry: for evidence, derive n_in, n_out from the task config ---
    if tc.kind == "evidence":
        n_in, _T = evidence_geometry(tc)
        tc.n_in, tc.n_out = n_in, 2
        n_out = 2
    else:
        n_in, n_out = tc.n_in, tc.n_out

    # --- Initial weights (scaled random init; recurrent diagonal zeroed) ---
    W_rec0 = (trc.w_gain / math.sqrt(n)) * torch.randn(n, n, generator=g)       # recurrent
    W_rec0.fill_diagonal_(0.0)                                                  # no self-connections
    W_in0 = (trc.w_gain / math.sqrt(n_in)) * torch.randn(n, n_in, generator=g) # input
    W_out0 = (1.0 / math.sqrt(n)) * torch.randn(n_out, n, generator=g)         # readout
    b_out = torch.zeros(n_out, device=dev)                                     # readout bias

    # --- Devices + synapses (W_rec and W_in are stored on device) ---
    def _make_syn_device(shape, seed_base):
        """Return one device, or a (pos, neg) pair when using differential pairs."""
        d_pos = _make_device(shape, dc, dev, seed_base)
        if sc.differential_pair and dc.kind != "ideal":
            d_neg = _make_device(shape, dc, dev, seed_base + 100)   # second device with a different seed
            return (d_pos, d_neg)
        return d_pos

    syn_rec = Synapse(_make_syn_device((n, n), trc.seed + 1),
                      sc.w_range, sc.writer, sc.verify_max_iter,
                      differential_pair=sc.differential_pair, auto_refresh=sc.auto_refresh)
    syn_in = Synapse(_make_syn_device((n, n_in), trc.seed + 2),
                     sc.w_range, sc.writer, sc.verify_max_iter,
                     differential_pair=sc.differential_pair, auto_refresh=sc.auto_refresh)
    syn_rec.init_weight(W_rec0.to(dev))                            # place initial weights on device
    syn_in.init_weight(W_in0.to(dev))

    # --- Readout: ideal (default) or on device (ablation) ---
    if trc.readout_on_device:
        syn_out = Synapse(_make_syn_device((n_out, n), trc.seed + 3),
                          sc.w_range, sc.writer, sc.verify_max_iter,
                          differential_pair=sc.differential_pair, auto_refresh=sc.auto_refresh)
        syn_out.init_weight(W_out0.to(dev))
        readout = syn_out
    else:
        readout = W_out0.to(dev)                                   # plain ideal tensor

    # --- Learning-signal feedback (fixed random matrix for the "random" variant) ---
    if trc.eprop_variant == "symmetric":
        B_fixed = None                                            # feedback = W_out (handled in network)
    elif trc.eprop_variant == "random":
        B_fixed = ((1.0 / math.sqrt(n)) * torch.randn(n_out, n, generator=g)).to(dev)
    else:
        raise ValueError(trc.eprop_variant)

    net = LSNN(nc, tc, syn_rec, syn_in, readout, b_out,
               variant=trc.eprop_variant, B_fixed=B_fixed, torch_device=dev)

    # --- Task data ---
    if tc.kind != "evidence":
        raise ValueError(f"Only 'evidence' task is supported, got: {tc.kind}")
    # An evidence task is a generator: each trial draws a fresh random cue sequence.
    task = {"kind": "evidence", "tc": tc,
            "gen": torch.Generator(device="cpu").manual_seed(tc.seed)}
    return net, task


def _apply_update(net, res, lr, readout_trainable):
    """Apply one gradient-descent step: dW = -lr * grad, written via each synapse's writer."""
    net.syn_rec.update(-lr * res["grad_rec"])                     # recurrent weights (on device)
    net.syn_in.update(-lr * res["grad_in"])                       # input weights (on device)
    if readout_trainable:
        if net.readout_is_device:
            net.readout.update(-lr * res["grad_out"])            # readout on device -> via writer
        else:
            net.readout.add_(-lr * res["grad_out"])              # ideal readout -> direct update
        net.b_out.add_(-lr * res["grad_b"])                      # bias is always ideal


def _pulses(net):
    """Total programming pulses written so far (an energy proxy)."""
    p = net.syn_rec.n_pulses_total + net.syn_in.n_pulses_total
    if net.readout_is_device:
        p += net.readout.n_pulses_total
    return p


def train(cfg, verbose=True):
    """Run the full training loop and return (net, history).

    history logs trial index, loss, accuracy (evidence), and cumulative pulses, sampled
    every log_every trials.
    """
    trc, tc = cfg.train, cfg.task
    dev = trc.torch_device
    net, task = build(cfg)
    lr = trc.lr
    history = {"trial": [], "loss": [], "acc": [], "pulses": []}
    acc_ma = None                              # accuracy moving average (evidence task only)

    for it in range(trc.n_trials):
        # Fresh random trial each time -> the network must genuinely learn, not memorize.
        X, Ystar, mask, label = make_evidence_trial(task["tc"], task["gen"], torch_device=dev)
        res = net.run_trial(X, Ystar, mask=mask, loss="classification")
        # Decision = the class with the largest summed output over the decision window.
        dec = int((res["y"] * mask[:, None]).sum(0).argmax().item())
        correct = float(dec == label)
        # Exponential moving average of accuracy (~50-trial window).
        acc_ma = correct if acc_ma is None else 0.98 * acc_ma + 0.02 * correct

        _apply_update(net, res, lr, trc.readout_trainable)       # write the weight updates

        if it % trc.log_every == 0 or it == trc.n_trials - 1:
            history["trial"].append(it)
            history["loss"].append(res["loss"])
            history["acc"].append(acc_ma if acc_ma is not None else float("nan"))
            history["pulses"].append(_pulses(net))
            if verbose:
                acc_s = f" | acc {acc_ma:.3f}" if acc_ma is not None else ""
                print(f"  trial {it:5d} | loss {res['loss']:.4f}{acc_s} | pulses {history['pulses'][-1]}")

    return net, history