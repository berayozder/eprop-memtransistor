"""
run_matrix.py
=============
THE MAIN EXPERIMENT DRIVER. Runs every condition across many random seeds and reports
each condition's final accuracy as mean +- 95% confidence interval (single-seed numbers
are too noisy to publish). Also saves a CSV and a three-panel figure.

The conditions together answer the project's questions:
  - ideal vs memtransistor vs differential-pair: how much does the device cost, and does
    the two-device mitigation recover it?
  - V_G sweep: how much does the gate voltage help resolution?
  - non-ideality ablation: which single flaw is responsible?
  - states sweep: is the loss dominated by coarse quantization?

Usage:
    python run_matrix.py                 # light/fast (small network, few trials)
    python run_matrix.py --full          # full scale (slow; run on your own machine)
    python run_matrix.py --plot-only     # redraw the figure from an existing CSV

CUSTOM DESIGN: this entire experimental methodology -- the specific set of
ablation conditions, the multi-seed mean+-95%CI statistical framework, and the
"states sweep" as a proxy for the quantization hypothesis -- is this project's
own experimental design for probing which non-ideality matters most. Neither
paper prescribes an ablation study; there is nothing to "reference" here, this
is original experiment design built around the two papers' models.
"""
from __future__ import annotations
import sys, copy, csv, math
import numpy as np
import torch

from config import ExperimentConfig
from train import train


# --------------------------- condition definitions ---------------------------
def base_config(full=False):
    """Return the base config shared by all conditions (task, network size, trials).

    `full` selects the publication-scale settings (larger network, longer task, more
    trials); otherwise a small/fast version for quick checks.
    """
    cfg = ExperimentConfig()
    cfg.task.kind = "evidence"
    cfg.neuron.adaptive_frac = 0.4
    cfg.train.lr = 0.02
    if full:
        cfg.task.n_cues = 7; cfg.task.cue_dur = 15; cfg.task.gap = 5
        cfg.task.delay = 50; cfg.task.decision = 30      # sequence length T = 220
        cfg.neuron.n_rec = 120; cfg.train.n_trials = 3000
    else:
        cfg.task.n_cues = 5; cfg.task.cue_dur = 10; cfg.task.gap = 4
        cfg.task.delay = 30; cfg.task.decision = 20      # T = 120
        cfg.neuron.n_rec = 80; cfg.train.n_trials = 800
    cfg.train.log_every = max(cfg.train.n_trials // 20, 1)
    return cfg


def apply(cfg, diff):
    """Return a deep copy of cfg with dotted-path overrides applied.

    Example: diff = {"device.kind": "ideal", "synapse.differential_pair": True}
    sets cfg.device.kind and cfg.synapse.differential_pair. Deep-copying keeps
    conditions fully independent of each other.
    """
    cfg = copy.deepcopy(cfg)
    for k, v in diff.items():
        obj, attr = k.rsplit(".", 1)          # split "device.kind" -> ("device", "kind")
        setattr(getattr(cfg, obj), attr, v)
    return cfg


def conditions(full=False):
    """Map condition name -> the config override that defines it.

    The base is the default memtransistor (fitted, asymmetric, single device).
    """
    C = {}
    C["ideal"]         = {"device.kind": "ideal"}                                          # perfect ceiling
    C["memtransistor"] = {"device.kind": "memtransistor"}                                  # realistic single device
    C["memtx (diff pair)"] = {"device.kind": "memtransistor", "synapse.differential_pair": True}  # mitigation
    # V_G sweep: does the gate voltage recover accuracy?
    for vg in [-50, -25, 0, 25, 50]:
        C[f"V_G={vg}"] = {"device.kind": "memtransistor", "device.V_G": float(vg)}
    # Non-ideality ablation: turn each flaw off one at a time.
    C["abl: asymmetry off"]   = {"device.kind": "memtransistor", "device.enable_asymmetry": False}
    C["abl: nonlinearity off"]= {"device.kind": "memtransistor", "device.enable_nonlinearity": False}
    C["abl: c2c off"]         = {"device.kind": "memtransistor", "device.enable_c2c": False}
    C["abl: d2d off"]         = {"device.kind": "memtransistor", "device.enable_d2d": False}
    # States sweep: smaller steps => more states. Does adding states recover the gap
    # (i.e. is the loss dominated by coarse quantization)?
    C["states x2"]  = {"device.kind": "memtransistor", "device.step_scale": 0.5}
    C["states x5"]  = {"device.kind": "memtransistor", "device.step_scale": 0.2}
    C["states x10"] = {"device.kind": "memtransistor", "device.step_scale": 0.1}
    return C


# --------------------------- running ---------------------------
def final_metric(history, last_frac=0.25):
    """Final accuracy = mean of the last 25% of logged accuracy values (reduces noise)."""
    acc = [a for a in history["acc"] if not math.isnan(a)]
    if not acc:
        return float("nan")
    k = max(1, int(len(acc) * last_frac))
    return float(np.mean(acc[-k:]))


def run_condition(cfg, seeds):
    """Run one condition across several seeds; return mean, 95% CI, and average pulses.

    Each seed sets both the weight-init seed and the task-data seed, giving one
    independent run per seed.
    """
    accs, pulses = [], []
    for s in seeds:
        c = copy.deepcopy(cfg); c.train.seed = s; c.task.seed = s
        _, h = train(c, verbose=False)
        accs.append(final_metric(h))
        pulses.append(h["pulses"][-1])
    accs = np.array(accs)
    mean = float(accs.mean())
    # Standard error of the mean; 95% CI ~ 1.96 * SEM. (0 if only one seed.)
    sem = accs.std(ddof=1) / math.sqrt(len(accs)) if len(accs) > 1 else 0.0
    ci95 = 1.96 * sem
    return dict(mean=float(mean), ci95=float(ci95), accs=accs.tolist(),
                pulses=float(np.mean(pulses)))


def run_matrix(full=False, n_seeds=5):
    """Run all conditions and print the results table; return the results dict."""
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


# --------------------------- outputs ---------------------------
def save_csv(results, fname="matrix_results.csv"):
    """Write the results table to CSV."""
    with open(fname, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["condition", "acc_mean", "ci95", "avg_pulses"])
        for name, r in results.items():
            w.writerow([name, f"{r['mean']:.4f}", f"{r['ci95']:.4f}", f"{r['pulses']:.1f}"])
    print(f"CSV: {fname}")


def plot(results, fname="matrix_figures.png"):
    """Draw the three-panel summary figure (V_G sweep, ablation, states sweep)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) V_G sweep
    vgs = [-50, -25, 0, 25, 50]
    m = [results[f"V_G={v}"]["mean"] for v in vgs]
    e = [results[f"V_G={v}"]["ci95"] for v in vgs]
    ax[0].errorbar(vgs, m, yerr=e, marker="o", capsize=4, label="memtransistor")
    ax[0].axhline(results["ideal"]["mean"], ls="--", c="green", label="ideal")
    ax[0].axhline(0.5, ls=":", c="gray", label="chance")
    ax[0].set_xlabel("V_G (V)"); ax[0].set_ylabel("accuracy (mean±CI95)")
    ax[0].set_title("V_G sweep (resolution)"); ax[0].legend(); ax[0].set_ylim(0.4, 1.0)

    # (b) non-ideality ablation + differential-pair mitigation
    names = ["ideal", "memtransistor", "memtx (diff pair)", "abl: asymmetry off",
             "abl: nonlinearity off", "abl: c2c off", "abl: d2d off"]
    labels = ["ideal", "memtx\n(full)", "diff\npair", "asym\noff", "nonlin\noff", "c2c\noff", "d2d\noff"]
    m = [results[n]["mean"] for n in names]; e = [results[n]["ci95"] for n in names]
    cols = ["green", "C3", "C1"] + ["C0"]*4
    ax[1].bar(range(len(names)), m, yerr=e, capsize=4, color=cols)
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].axhline(0.5, ls=":", c="gray")
    ax[1].set_ylabel("accuracy (mean±CI95)"); ax[1].set_title("Ablation & diff-pair mitigation"); ax[1].set_ylim(0.4, 1.0)

    # (c) states (quantization) sweep
    snames = ["memtransistor", "states x2", "states x5", "states x10"]
    slabels = ["~4\n(full)", "~7\n(x2)", "~18\n(x5)", "~37\n(x10)"]
    m = [results[n]["mean"] for n in snames]; e = [results[n]["ci95"] for n in snames]
    ax[2].errorbar(range(len(snames)), m, yerr=e, marker="s", capsize=4, color="C3", label="memtransistor")
    ax[2].axhline(results["ideal"]["mean"], ls="--", c="green", label="ideal")
    ax[2].axhline(0.5, ls=":", c="gray", label="chance")
    ax[2].set_xticks(range(len(snames))); ax[2].set_xticklabels(slabels, fontsize=8)
    ax[2].set_xlabel("effective number of states"); ax[2].set_ylabel("accuracy (mean±CI95)")
    ax[2].set_title("States (quantization)"); ax[2].legend(); ax[2].set_ylim(0.4, 1.0)

    fig.tight_layout(); fig.savefig(fname, dpi=120)
    print(f"figure: {fname}")


def load_results_csv(fname="matrix_results.csv"):
    """Reload a results dict from a saved CSV (to redraw the figure without re-running)."""
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
        # Redraw the figure from an existing matrix_results.csv (no re-running).
        results = load_results_csv()
        plot(results)
    else:
        full = "--full" in sys.argv
        n_seeds = 10 if full else 5
        torch.manual_seed(0)
        results = run_matrix(full=full, n_seeds=n_seeds)
        save_csv(results)
        plot(results)