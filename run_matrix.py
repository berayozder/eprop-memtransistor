"""
run_matrix.py
-------------
Multi-seed experiment harness. Runs each experimental condition across N seeds
and reports the mean final accuracy ± 95% Confidence Interval (CI) for reliable reporting.

Headline outputs:
  - V_G sweep: how gate-dependent switching ratio/resolution affects accuracy.
  - Non-ideality ablation: disables each physical non-ideality one-by-one to find critical bottlenecks.
  - State resolution sweep: scales step sizes to test the quantization resolution limit.

Usage:
    python run_matrix.py                 # Fast/light mode for quick visualization
    python run_matrix.py --full          # Full scale run (slower, matches paper settings)
"""
from __future__ import annotations
import sys
import copy
import csv
import math
import numpy as np
import torch

from config import ExperimentConfig
from train import train


# --------------------------- Condition Definitions ---------------------------
def base_config(full=False):
    cfg = ExperimentConfig()
    cfg.task.kind = "evidence"
    cfg.neuron.adaptive_frac = 0.4
    cfg.train.lr = 0.02
    if full:
        cfg.task.n_cues = 7
        cfg.task.cue_dur = 15
        cfg.task.gap = 5
        cfg.task.delay = 50
        cfg.task.decision = 30      # T = 220
        cfg.neuron.n_rec = 120
        cfg.train.n_trials = 3000
    else:
        cfg.task.n_cues = 5
        cfg.task.cue_dur = 10
        cfg.task.gap = 4
        cfg.task.delay = 30
        cfg.task.decision = 20      # T = 120
        cfg.neuron.n_rec = 80
        cfg.train.n_trials = 800
    cfg.train.log_every = max(cfg.train.n_trials // 20, 1)
    return cfg


def apply(cfg, diff):
    cfg = copy.deepcopy(cfg)
    for k, v in diff.items():
        obj, attr = k.rsplit(".", 1)
        setattr(getattr(cfg, obj), attr, v)
    return cfg


def conditions(full=False):
    """Mapping: name -> config-diff. Base is memtransistor default (fitted, asymmetric)."""
    C = {}
    C["ideal"]         = {"device.kind": "ideal"}
    C["memtransistor"] = {"device.kind": "memtransistor"}
    
    # V_G sweep
    for vg in [-50, -25, 0, 25, 50]:
        C[f"V_G={vg}"] = {"device.kind": "memtransistor", "device.V_G": float(vg)}
        
    # Non-ideality ablation (disables one key non-ideality at a time)
    C["abl: asymmetry off"]    = {"device.kind": "memtransistor", "device.enable_asymmetry": False}
    C["abl: nonlinearity off"] = {"device.kind": "memtransistor", "device.enable_nonlinearity": False}
    C["abl: c2c off"]         = {"device.kind": "memtransistor", "device.enable_c2c": False}
    C["abl: d2d off"]         = {"device.kind": "memtransistor", "device.enable_d2d": False}
    
    # State count (quantization resolution) sweep: tests if performance is resolution-limited
    C["states x2"]  = {"device.kind": "memtransistor", "device.step_scale": 0.5}
    C["states x5"]  = {"device.kind": "memtransistor", "device.step_scale": 0.2}
    C["states x10"] = {"device.kind": "memtransistor", "device.step_scale": 0.1}
    return C


# --------------------------- Execution ---------------------------
def final_metric(history, last_frac=0.25):
    """Averages the last fraction of recorded trial accuracy to filter out single-point noise."""
    acc = [a for a in history["acc"] if not math.isnan(a)]
    if not acc:
        return float("nan")
    k = max(1, int(len(acc) * last_frac))
    return float(np.mean(acc[-k:]))


def run_condition(cfg, seeds):
    accs, pulses = [], []
    for s in seeds:
        c = copy.deepcopy(cfg)
        c.train.seed = s
        c.task.seed = s
        _, h = train(c, verbose=False)
        accs.append(final_metric(h))
        pulses.append(h["pulses"][-1])
    accs = np.array(accs)
    mean = accs.mean()
    sem = accs.std(ddof=1) / math.sqrt(len(accs)) if len(accs) > 1 else 0.0
    ci95 = 1.96 * sem
    return dict(mean=float(mean), ci95=float(ci95), accs=accs.tolist(),
                pulses=float(np.mean(pulses)))


def run_matrix(full=False, n_seeds=5):
    base = base_config(full)
    seeds = list(range(n_seeds))
    C = conditions(full)
    results = {}
    print(f"{'condition':24s} {'acc mean':>9s} {'+-CI95':>7s} {'pulses':>8s}  (n={n_seeds} seeds)")
    print("-" * 56)
    for name, diff in C.items():
        r = run_condition(apply(base, diff), seeds)
        results[name] = r
        print(f"{name:24s} {r['mean']:>9.3f} {r['ci95']:>7.3f} {r['pulses']:>8.0f}")
    return results


# --------------------------- Outputs / Plotting ---------------------------
def save_csv(results, fname="matrix_results.csv"):
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "acc_mean", "ci95", "avg_pulses"])
        for name, r in results.items():
            w.writerow([name, f"{r['mean']:.4f}", f"{r['ci95']:.4f}", f"{r['pulses']:.1f}"])
    print(f"CSV saved to: {fname}")


def plot(results, fname="matrix_figures.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) V_G Sweep
    vgs = [-50, -25, 0, 25, 50]
    m = [results[f"V_G={v}"]["mean"] for v in vgs]
    e = [results[f"V_G={v}"]["ci95"] for v in vgs]
    ax[0].errorbar(vgs, m, yerr=e, marker="o", capsize=4, label="memtransistor")
    ax[0].axhline(results["ideal"]["mean"], ls="--", c="green", label="ideal")
    ax[0].axhline(0.5, ls=":", c="gray", label="chance")
    ax[0].set_xlabel("V_G (V)")
    ax[0].set_ylabel("accuracy (mean±CI95)")
    ax[0].set_title("V_G Sweep (Resolution)")
    ax[0].legend()
    ax[0].set_ylim(0.4, 1.0)

    # (b) Ablation Study
    names = ["ideal", "memtransistor", "abl: asymmetry off", "abl: nonlinearity off",
             "abl: c2c off", "abl: d2d off"]
    labels = ["ideal", "memtx\n(full)", "asymmetry\noff", "nonlin\noff", "c2c\noff", "d2d\noff"]
    m = [results[n]["mean"] for n in names]
    e = [results[n]["ci95"] for n in names]
    cols = ["green", "C3"] + ["C0"]*4
    ax[1].bar(range(len(names)), m, yerr=e, capsize=4, color=cols)
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].axhline(0.5, ls=":", c="gray")
    ax[1].set_ylabel("accuracy (mean±CI95)")
    ax[1].set_title("Non-Ideality Ablation")
    ax[1].set_ylim(0.4, 1.0)

    # (c) State Count (Quantization) Sweep
    snames = ["memtransistor", "states x2", "states x5", "states x10"]
    slabels = ["~4\n(base)", "~7\n(x2)", "~18\n(x5)", "~37\n(x10)"]
    m = [results[n]["mean"] for n in snames]
    e = [results[n]["ci95"] for n in snames]
    ax[2].errorbar(range(len(snames)), m, yerr=e, marker="s", capsize=4, color="C3", label="memtransistor")
    ax[2].axhline(results["ideal"]["mean"], ls="--", c="green", label="ideal")
    ax[2].axhline(0.5, ls=":", c="gray", label="chance")
    ax[2].set_xticks(range(len(snames)))
    ax[2].set_xticklabels(slabels, fontsize=8)
    ax[2].set_xlabel("Effective States")
    ax[2].set_ylabel("accuracy (mean±CI95)")
    ax[2].set_title("State Count (Quantization)")
    ax[2].legend()
    ax[2].set_ylim(0.4, 1.0)

    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print(f"Figure saved to: {fname}")


def load_results_csv(fname="matrix_results.csv"):
    """Loads results dict from a previously saved CSV file to regenerate plots quickly."""
    results = {}
    with open(fname) as f:
        r = csv.DictReader(f)
        for row in r:
            results[row["condition"]] = {
                "mean": float(row["acc_mean"]),
                "ci95": float(row["ci95"]),
                "pulses": float(row["avg_pulses"]),
            }
    return results


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        # Generate plot from existing matrix_results.csv without re-running simulations
        results = load_results_csv()
        plot(results)
    else:
        full = "--full" in sys.argv
        n_seeds = 10 if full else 5
        torch.manual_seed(0)
        results = run_matrix(full=full, n_seeds=n_seeds)
        save_csv(results)
        plot(results)