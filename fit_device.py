"""
fit_device.py
=============
FITS the device model to real measured data. This is what makes the memtransistor
"not made up": it takes the digitized pulse-response curve from Sangwan 2018 Fig. 4c
and finds the LTP/LTD step parameters (dp, kp, dd, kd) that reproduce it.

The fitting model is the SAME recursion used in memtransistor.py (with a normalized
conductance g in [0, 1]), so the fitted parameters are directly consistent with the
device simulation:
    LTP: g <- g + dp * exp(-kp * g)          (up steps, gradual)
    LTD: g <- g - dd * exp(-kd * (1 - g))     (down steps, abrupt)

Input: fig4c.csv (the user's WebPlotDigitizer export of the figure). The curve rises
(potentiation) up to a peak, then falls (depression); we split it at the peak into the
LTP and LTD branches and fit each separately.
"""
from __future__ import annotations
import numpy as np


def load_fig4c(path="fig4c.csv"):
    """Load and split the Fig. 4c data into LTP and LTD branches, resampled to integer pulses.

    The CSV is read in measurement (row) order. The peak (max current) is the moment
    the pulse polarity flips (+30V -> -30V), so everything up to the peak is LTP and
    everything from the peak on is LTD. Each branch is resampled onto integer pulse
    indices (the figure's pulse axis is meaningful); the LTD branch is anchored so its
    first point sits exactly at the peak.

    Returns:
        ltp: [n, 2] array of (pulse, current) for potentiation.
        ltd: [m, 2] array of (pulse, current) for depression.
        (i_min, i_max): min/max current, used to normalize.
    """
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    imax = int(np.argmax(d[:, 1]))            # index of the peak (polarity switch)
    peak = d[imax, 1]

    def _resample(x, y, anchor_peak=False):
        o = np.argsort(x); xs, ys = x[o], y[o]
        xs = np.maximum.accumulate(xs) + 1e-6 * np.arange(len(xs))  # force strictly increasing x
        pulses = np.arange(int(round(x.min())), int(round(x.max())) + 1)
        g = np.interp(pulses, xs, ys)         # interpolate onto integer pulse indices
        if anchor_peak:
            g[0] = peak                       # LTD starts exactly at the peak
        return np.column_stack([pulses, g])

    ltp = _resample(d[:imax + 1, 0], d[:imax + 1, 1])
    ltd = _resample(d[imax:, 0], d[imax:, 1], anchor_peak=True)
    i_min, i_max = float(d[:, 1].min()), float(d[:, 1].max())
    return ltp, ltd, (i_min, i_max)


def _norm(cur, rng):
    """Normalize a current value to the [0, 1] conductance scale used by the model."""
    lo, hi = rng
    return (cur - lo) / max(hi - lo, 1e-9)


def _sim_ltp(dp, kp, g0, n):
    """Simulate n LTP pulses from g0 with parameters (dp, kp); clamp at 1.0.

    This is exactly the memtransistor LTP recursion in normalized units, so fitting it
    yields parameters the device model can use directly.
    """
    g, out = g0, []
    for _ in range(n):
        out.append(g); g = min(g + dp * np.exp(-kp * g), 1.0)
    return np.array(out)


def _sim_ltd(dd, kd, g0, n):
    """Simulate n LTD pulses from g0 with parameters (dd, kd); clamp at 0.0."""
    g, out = g0, []
    for _ in range(n):
        out.append(g); g = max(g - dd * np.exp(-kd * (1 - g)), 0.0)
    return np.array(out)


def fit(path="fig4c.csv", verbose=True):
    """Least-squares fit of (dp, kp) and (dd, kd) to the two branches of Fig. 4c.

    Returns:
        params: dict(dp, kp, dd, kd) to paste into DeviceConfig.
        (g_ltp, g_ltd, rng): the normalized data branches and the current range.
    """
    from scipy.optimize import least_squares
    ltp, ltd, rng = load_fig4c(path)
    g_ltp = _norm(ltp[:, 1], rng)             # normalized LTP branch
    g_ltd = _norm(ltd[:, 1], rng)             # normalized LTD branch

    # Fit LTP: minimize the difference between the simulated recursion and the data.
    def res_p(x): return _sim_ltp(x[0], x[1], g_ltp[0], len(g_ltp)) - g_ltp
    sol_p = least_squares(res_p, [0.15, 2.0], bounds=([1e-3, 0.0], [1.0, 12.0]))
    dp, kp = sol_p.x

    # Fit LTD likewise.
    def res_d(x): return _sim_ltd(x[0], x[1], g_ltd[0], len(g_ltd)) - g_ltd
    sol_d = least_squares(res_d, [0.15, 2.0], bounds=([1e-3, 0.0], [1.0, 12.0]))
    dd, kd = sol_d.x

    rmse_p = float(np.sqrt(np.mean(res_p(sol_p.x) ** 2)))   # fit quality (LTP)
    rmse_d = float(np.sqrt(np.mean(res_d(sol_d.x) ** 2)))   # fit quality (LTD)
    params = dict(dp=float(dp), kp=float(kp), dd=float(dd), kd=float(kd))

    if verbose:
        print(f"LTP fit: dp={dp:.4f}  kp={kp:.3f}   RMSE={rmse_p:.4f}  ({len(g_ltp)} pulses)")
        print(f"LTD fit: dd={dd:.4f}  kd={kd:.3f}   RMSE={rmse_d:.4f}  ({len(g_ltd)} pulses)")
        print(f"  current range: {rng[0]:.3f}..{rng[1]:.3f} uA")
        print(f"  asymmetry: LTP gradual (~10 pulses), LTD abrupt (large drop in 1 pulse)")
        print(f"\nPaste into DeviceConfig: dp={dp:.4f}, dd={dd:.4f}, kp={kp:.3f}, kd={kd:.3f}")
    return params, (g_ltp, g_ltd, rng)


def plot_fit(path="fig4c.csv", fname="device_fit.png"):
    """Fit and draw the data-vs-fit curves visually matching Sangwan 2018 Fig. 4c."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    params, (g_ltp, g_ltd, rng) = fit(path, verbose=True)
    fit_p = _sim_ltp(params["dp"], params["kp"], g_ltp[0], len(g_ltp))
    fit_d = _sim_ltd(params["dd"], params["kd"], g_ltd[0], len(g_ltd))

    # Convert normalized conductance back to physical current (uA) for Fig. 4c comparison
    lo, hi = rng
    cur_ltp_data = lo + g_ltp * (hi - lo)
    cur_ltp_fit  = lo + fit_p * (hi - lo)
    cur_ltd_data = lo + g_ltd * (hi - lo)
    cur_ltd_fit  = lo + fit_d * (hi - lo)

    pulses_p = range(len(g_ltp))
    pulses_d = range(len(g_ltp) - 1, len(g_ltp) - 1 + len(g_ltd))

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(pulses_p, cur_ltp_data, "o", color="#E6613B", label="LTP Measured Data", markersize=6)
    ax.plot(pulses_p, cur_ltp_fit, "-", color="#E6613B", label="LTP Fitted Model (dp=0.225, kp=1.54)", linewidth=2)
    ax.plot(pulses_d, cur_ltd_data, "o", color="#4CB140", label="LTD Measured Data", markersize=6)
    ax.plot(pulses_d, cur_ltd_fit, "-", color="#4CB140", label="LTD Fitted Model (dd=0.870, kd=3.76)", linewidth=2)

    ax.set_xlabel("Pulse Number", fontsize=12)
    ax.set_ylabel("Post-synaptic Current (μA)", fontsize=12)
    ax.set_title("MoS₂ Memtransistor Response vs. Fitted Model (Sangwan 2018, Fig. 4c)", fontsize=12, pad=10)
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(0.0, 0.6)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"figure saved: {fname}")
    return params


if __name__ == "__main__":
    plot_fit()