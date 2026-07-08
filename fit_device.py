"""
fit_device.py
-------------
Fits the state-dependent LTP/LTD step parameters (dp, kp, dd, kd) in memtransistor.py
to the experimental pulse-response data from Sangwan 2018 (Fig. 4c).

Model (identical recursion to memtransistor.py, normalized conductance g in [0, 1]):
    LTP: g <- g + dp*exp(-kp*g)
    LTD: g <- g - dd*exp(-kd*(1-g))

Data: fig4c.csv (WebPlotDigitizer output from the user). It splits at the peak into
distinct LTP (rising) and LTD (falling) branches.
"""
from __future__ import annotations
import numpy as np


def load_fig4c(path="fig4c.csv"):
    """
    Reads the CSV in measurement order; splits at the peak into LTP and LTD branches;
    resamples each branch to integer pulse counts (the pulse axis of the figure is meaningful).
    LTD pulse-0 is anchored to the peak value (combining the red and green curves).
    Returns: ltp [n,2]=(pulse, current), ltd [m,2]=(pulse, current), (i_min, i_max)
    """
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    imax = int(np.argmax(d[:, 1]))
    peak = d[imax, 1]

    def _resample(x, y, anchor_peak=False):
        o = np.argsort(x)
        xs, ys = x[o], y[o]
        xs = np.maximum.accumulate(xs) + 1e-6 * np.arange(len(xs))  # Ensure strictly increasing
        pulses = np.arange(int(round(x.min())), int(round(x.max())) + 1)
        g = np.interp(pulses, xs, ys)
        if anchor_peak:
            g[0] = peak
        return np.column_stack([pulses, g])

    ltp = _resample(d[:imax + 1, 0], d[:imax + 1, 1])
    ltd = _resample(d[imax:, 0], d[imax:, 1], anchor_peak=True)
    i_min, i_max = float(d[:, 1].min()), float(d[:, 1].max())
    return ltp, ltd, (i_min, i_max)


def _norm(cur, rng):
    lo, hi = rng
    return (cur - lo) / max(hi - lo, 1e-9)


def _sim_ltp(dp, kp, g0, n):
    g, out = g0, []
    for _ in range(n):
        out.append(g)
        g = min(g + dp * np.exp(-kp * g), 1.0)
    return np.array(out)


def _sim_ltd(dd, kd, g0, n):
    g, out = g0, []
    for _ in range(n):
        out.append(g)
        g = max(g - dd * np.exp(-kd * (1 - g)), 0.0)
    return np.array(out)


def fit(path="fig4c.csv", verbose=True):
    from scipy.optimize import least_squares
    ltp, ltd, rng = load_fig4c(path)
    g_ltp = _norm(ltp[:, 1], rng)
    g_ltd = _norm(ltd[:, 1], rng)

    def res_p(x):
        return _sim_ltp(x[0], x[1], g_ltp[0], len(g_ltp)) - g_ltp
    sol_p = least_squares(res_p, [0.15, 2.0], bounds=([1e-3, 0.0], [1.0, 12.0]))
    dp, kp = sol_p.x

    def res_d(x):
        return _sim_ltd(x[0], x[1], g_ltd[0], len(g_ltd)) - g_ltd
    sol_d = least_squares(res_d, [0.15, 2.0], bounds=([1e-3, 0.0], [1.0, 12.0]))
    dd, kd = sol_d.x

    rmse_p = float(np.sqrt(np.mean(res_p(sol_p.x) ** 2)))
    rmse_d = float(np.sqrt(np.mean(res_d(sol_d.x) ** 2)))
    params = dict(dp=float(dp), kp=float(kp), dd=float(dd), kd=float(kd))

    if verbose:
        print(f"LTP fit: dp={dp:.4f}  kp={kp:.3f}   RMSE={rmse_p:.4f}  ({len(g_ltp)} pulses)")
        print(f"LTD fit: dd={dd:.4f}  kd={kd:.3f}   RMSE={rmse_d:.4f}  ({len(g_ltd)} pulses)")
        print(f"  Current range: {rng[0]:.3f}..{rng[1]:.3f} uA")
        print(f"  Asymmetry: Gradual LTP (~10 pulses), sudden LTD (large drop in 1st pulse)")
        print(f"\nUpdate DeviceConfig with: dp={dp:.4f}, dd={dd:.4f}, kp={kp:.3f}, kd={kd:.3f}")
    return params, (g_ltp, g_ltd, rng)


def plot_fit(path="fig4c.csv", fname="device_fit.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    params, (g_ltp, g_ltd, rng) = fit(path, verbose=True)
    fit_p = _sim_ltp(params["dp"], params["kp"], g_ltp[0], len(g_ltp))
    fit_d = _sim_ltd(params["dd"], params["kd"], g_ltd[0], len(g_ltd))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(range(len(g_ltp)), g_ltp, "o", color="C0", label="LTP data")
    ax.plot(range(len(g_ltp)), fit_p, "-", color="C0", label="LTP fit")
    ax.plot(range(len(g_ltd)), g_ltd, "s", color="C3", label="LTD data")
    ax.plot(range(len(g_ltd)), fit_d, "--", color="C3", label="LTD fit")
    ax.set_xlabel("within-branch pulse #")
    ax.set_ylabel("normalized conductance g")
    ax.set_title("Fig. 4c fit (gradual LTP vs sudden LTD asymmetry)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print(f"Figure saved to: {fname}")
    return params


if __name__ == "__main__":
    plot_fit()