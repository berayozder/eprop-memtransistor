"""
run_experiment.py
=================
A SIMPLE HELPER experiment: train the ideal baseline vs the memtransistor on the
evidence task (same e-prop, same everything else) and plot their learning curves
(accuracy and loss over trials). This is a quick single-seed sanity view; the full
multi-seed study lives in run_matrix.py.

Usage:
    python run_experiment.py
"""
from __future__ import annotations
import copy
import torch

from config import ExperimentConfig
from train import train


def evidence_config(light=True):
    """Return an evidence-task config. `light` uses a small/fast setup for quick figures."""
    cfg = ExperimentConfig()
    cfg.task.kind = "evidence"
    if light:   # smaller network / shorter task for fast plots
        cfg.task.n_cues = 5; cfg.task.cue_dur = 10; cfg.task.gap = 4
        cfg.task.delay = 30; cfg.task.decision = 20        # sequence length T = 120
        cfg.neuron.n_rec = 80
        cfg.train.n_trials = 800
    cfg.neuron.adaptive_frac = 0.4
    cfg.train.lr = 0.02
    cfg.train.log_every = 40
    return cfg


def run_ideal_vs_memtransistor(light=True):
    """Train both the ideal and the memtransistor arms; print and return their histories."""
    base = evidence_config(light)
    ci = copy.deepcopy(base); ci.device.kind = "ideal"
    print("[ideal baseline]");             _, h_i = train(ci, verbose=False)
    cm = copy.deepcopy(base); cm.device.kind = "memtransistor"
    print("[memtransistor + accumulate]"); _, h_m = train(cm, verbose=False)
    print(f"ideal   final acc {h_i['acc'][-1]:.3f}")
    print(f"memtx   final acc {h_m['acc'][-1]:.3f} | device pulses {h_m['pulses'][-1]}")
    return h_i, h_m


def plot(h_i, h_m, fname="evidence_curves.png"):
    """Plot accuracy and loss learning curves for the two arms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h_i["trial"], h_i["acc"], label="ideal baseline")
    ax[0].plot(h_m["trial"], h_m["acc"], label="memtransistor")
    ax[0].axhline(0.5, ls=":", c="gray", label="chance (0.5)")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("accuracy (moving avg)")
    ax[0].set_ylim(0.4, 1.0); ax[0].legend(); ax[0].set_title("Evidence accumulation")

    ax[1].plot(h_i["trial"], h_i["loss"], label="ideal")
    ax[1].plot(h_m["trial"], h_m["loss"], label="memtransistor")
    ax[1].set_xlabel("trial"); ax[1].set_ylabel("cross-entropy loss")
    ax[1].set_yscale("log"); ax[1].legend(); ax[1].set_title("Loss")

    fig.tight_layout(); fig.savefig(fname, dpi=120)
    print(f"figure saved: {fname}")


if __name__ == "__main__":
    torch.manual_seed(0)
    h_i, h_m = run_ideal_vs_memtransistor(light=True)
    plot(h_i, h_m)