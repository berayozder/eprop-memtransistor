"""
run_experiment.py
-----------------
Ideal baseline vs memtransistor karsilastirmasi (ayni gorev, ayni e-prop).
Ciktilar: ogrenme egrileri + son cikti izleri figuru.

Kullanim:
    python run_experiment.py
Ekstra deneyler (V_G suprumu, writer=direct, ablation) alttaki main'de acilir.
"""
from __future__ import annotations
import copy
import torch

from config import ExperimentConfig
from train import train, build


def run_ideal_vs_memtransistor():
    base = ExperimentConfig()

    # --- 1) ideal baseline ---
    cfg_ideal = copy.deepcopy(base)
    cfg_ideal.device.kind = "ideal"
    print("[ideal baseline]")
    _, h_ideal = train(cfg_ideal)

    # --- 2) memtransistor (accumulate writer) ---
    cfg_mt = copy.deepcopy(base)
    cfg_mt.device.kind = "memtransistor"
    print("[memtransistor + accumulate]")
    net_mt, h_mt = train(cfg_mt)

    return (cfg_ideal, h_ideal), (cfg_mt, h_mt, net_mt)


def plot(results, fname="learning_curves.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (cfg_i, h_i), (cfg_m, h_m, net_m) = results
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(h_i["trial"], h_i["loss"], label="ideal baseline")
    ax[0].plot(h_m["trial"], h_m["loss"], label="memtransistor")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("MSE loss"); ax[0].set_yscale("log")
    ax[0].legend(); ax[0].set_title("Ogrenme egrisi")

    # memtransistor son cikti vs hedef
    _, X, Y = build(cfg_m)
    res = net_m.run_trial(X, Y, accumulate_grads=False)
    y = res["y"].detach().cpu().numpy()[:, 0]
    ax[1].plot(Y.cpu().numpy()[:, 0], "k--", label="hedef")
    ax[1].plot(y, label="memtransistor cikti")
    ax[1].set_xlabel("t (ms)"); ax[1].set_ylabel("y"); ax[1].legend()
    ax[1].set_title("Pattern generation (memtransistor)")

    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print(f"figur kaydedildi: {fname}")


if __name__ == "__main__":
    torch.manual_seed(0)
    results = run_ideal_vs_memtransistor()
    plot(results)