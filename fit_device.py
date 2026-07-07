"""
fit_device.py
-------------
Fits the state-dependent LTP/LTD conductance step parameters (dp, kp, dd, kd)
in memtransistor.py to the experimental pulse-response data from Sangwan 2018 (Fig. 4c).

Model (identical recursion to memtransistor.py, normalized conductance g in [0,1]):
    LTP: g <- g + dp*exp(-kp*g)
    LTD: g <- g - dd*exp(-kd*(1-g))

>>> IMPORTANT <<<
The REPRESENTATIVE_* arrays below represent representative data points extracted
visually from Fig. 4c (saturation around 8-10 pulses, current range ~0.05-0.55 uA,
LTD slightly faster than LTP showing asymmetry) and serve as a placeholder.
For pixel-precise fitting, digitize Fig. 4c (e.g. using WebPlotDigitizer),
save a two-column CSV (pulse_index, current_uA), and load it via load_csv(...).
"""
from __future__ import annotations
import numpy as np


# --- REPRESENTATIVE placeholder (replace with actual digitized CSV) ----------
# LTP branch: +30V programming pulses, current ~0.05 -> ~0.55 uA with saturation
REPRESENTATIVE_LTP = np.array([
    [0, 0.05], [1, 0.19], [2, 0.30], [3, 0.37], [4, 0.42], [5, 0.46],
    [6, 0.49], [7, 0.51], [8, 0.525], [9, 0.535], [10, 0.54],
])
# LTD branch: -30V programming pulses, current ~0.54 -> ~0.06 uA, slightly faster than LTP (asymmetry)
REPRESENTATIVE_LTD = np.array([
    [0, 0.54], [1, 0.38], [2, 0.27], [3, 0.20], [4, 0.15], [5, 0.12],
    [6, 0.10], [7, 0.085], [8, 0.075], [9, 0.068], [10, 0.06],
])


def load_csv(path):
    """Loads a two-column CSV containing (pulse_index, current)."""
    return np.loadtxt(path, delimiter=",", skiprows=1)


def _normalize(ltp, ltd):
    """Normalizes the current values to a shared [0, 1] conductance scale using the absolute min/max range of LTP."""
    i_min = min(ltp[:, 1].min(), ltd[:, 1].min())
    i_max = max(ltp[:, 1].max(), ltd[:, 1].max())
    span = max(i_max - i_min, 1e-9)
    g_ltp = (ltp[:, 1] - i_min) / span
    g_ltd = (ltd[:, 1] - i_min) / span
    return g_ltp, g_ltd, (i_min, i_max)


def _sim_ltp(dp, kp, g0, n):
    g, out = g0, []
    for _ in range(n):
        out.append(g)
        g = min(g + dp * np.exp(-kp * g), 1.0)
    out.append(g)
    return np.array(out[:n])


def _sim_ltd(dd, kd, g0, n):
    g, out = g0, []
    for _ in range(n):
        out.append(g)
        g = max(g - dd * np.exp(-kd * (1 - g)), 0.0)
    out.append(g)
    return np.array(out[:n])


def fit(ltp=REPRESENTATIVE_LTP, ltd=REPRESENTATIVE_LTD, verbose=True):
    from scipy.optimize import least_squares

    g_ltp, g_ltd, cur_range = _normalize(ltp, ltd)
    n_p, n_d = len(g_ltp), len(g_ltd)

    # LTP fitting: optimizing [dp, kp] with g0 = initial data point
    def res_p(x):
        return _sim_ltp(x[0], x[1], g_ltp[0], n_p) - g_ltp
    sol_p = least_squares(res_p, x0=[0.15, 2.0],
                          bounds=([1e-3, 0.0], [1.0, 10.0]))
    dp, kp = sol_p.x

    # LTD fitting: optimizing [dd, kd] with g0 = initial (saturated) point
    def res_d(x):
        return _sim_ltd(x[0], x[1], g_ltd[0], n_d) - g_ltd
    sol_d = least_squares(res_d, x0=[0.15, 2.0],
                          bounds=([1e-3, 0.0], [1.0, 10.0]))
    dd, kd = sol_d.x

    rmse_p = float(np.sqrt(np.mean(res_p(sol_p.x) ** 2)))
    rmse_d = float(np.sqrt(np.mean(res_d(sol_d.x) ** 2)))

    params = dict(dp=float(dp), kp=float(kp), dd=float(dd), kd=float(kd))
    if verbose:
        print("Fitting Results (representative Fig. 4c data):")
        print(f"  dp={dp:.4f}  kp={kp:.3f}   (LTP, RMSE={rmse_p:.4f})")
        print(f"  dd={dd:.4f}  kd={kd:.3f}   (LTD, RMSE={rmse_d:.4f})")
        print(f"  Current range (uA): {cur_range[0]:.3f}..{cur_range[1]:.3f}")
        print("\nPaste parameters into DeviceConfig:")
        print(f"  dp={dp:.4f}, dd={dd:.4f}, kp={kp:.3f}, kd={kd:.3f}")
    return params, (g_ltp, g_ltd)


def plot_fit(fname="device_fit.png", ltp=REPRESENTATIVE_LTP, ltd=REPRESENTATIVE_LTD):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params, (g_ltp, g_ltd) = fit(ltp, ltd, verbose=True)
    n_p, n_d = len(g_ltp), len(g_ltd)
    fit_p = _sim_ltp(params["dp"], params["kp"], g_ltp[0], n_p)
    fit_d = _sim_ltd(params["dd"], params["kd"], g_ltd[0], n_d)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(n_p), g_ltp, "o", color="C0", label="LTP data (representative)")
    ax.plot(range(n_p), fit_p, "-", color="C0", label="LTP fit")
    ax.plot(range(n_d), g_ltd, "s", color="C3", label="LTD data (representative)")
    ax.plot(range(n_d), fit_d, "-", color="C3", label="LTD fit")
    ax.set_xlabel("pulse #")
    ax.set_ylabel("normalized conductance g")
    ax.set_title("Memtransistor Pulse-Response Fit (Fig. 4c)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print(f"Figure saved to: {fname}")
    return params


if __name__ == "__main__":
    plot_fit()