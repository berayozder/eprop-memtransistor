"""
physics_vg_check.py
====================
CUSTOM DESIGN, ENTIRE FILE: this is original exploratory research code, not a
transcription of either paper. It reduces Sangwan 2018's transport equations to a
deliberately simplified single-branch model (a project-specific simplification
choice, see EQUATIONS/SIMPLIFICATIONS below), fits it with an original multi-start
+ bound-free-proxy methodology, and uses it to test (not confirm) an assumption in
this project's OWN memtransistor.py heuristic. Nothing here is meant to be cited
back to Sangwan 2018 as if it were the paper's own model.

STANDALONE RESEARCH CHECK (not wired into the simulator): does the memtransistor
paper's own transport-physics equations (Sangwan 2018, Methods) predict the same
V_G -> resolution scaling law that memtransistor.py's `gate_gain()` heuristic
assumes, or something qualitatively different?

THE HYPOTHESIS UNDER TEST
  `gate_gain()` rescales the usable conductance WINDOW with V_G and deliberately
  keeps the per-pulse STEP SIZE (dp/dd/kp/kd) fixed -- i.e. it assumes V_G only
  ever adds more distinguishable states, never changes how big one pulse's effect
  is. But the paper's drain-current equation has an explicit gate-voltage term
  multiplying I_D directly, and the state update is dw/dt = mu*R_on*I_D(t)*F(w)
  (Eq. 16) -- i.e. V_G rescales the *rate* w moves per pulse, which is a STEP-SIZE
  effect, not a window-shape effect. This script derives that prediction directly
  from the paper's own equations (fit once at V_G=0V, the actual measurement
  condition of Fig. 4c/4d -- see memtransistor.py's Phase-0 fix) and compares it
  to the current heuristic.

EQUATIONS IMPLEMENTED (Sangwan 2018, Methods; general/unified form, i.e. ONE
formula for both pulse polarities -- not the separate V_D>=0/V_D<0 branches of
Eqs. 18-22, which would double the free parameters for likely marginal gain here):
  F(w)          = 1 - [(w-0.5)^2 + 0.75]^p                      (Eq. 17, p=4)
  Phi_b(w)      = phi_b0 - k_phi * sqrt(w)                      (Eq. 2, image-charge
                  term only -- see SIMPLIFICATIONS below)
  I_D(w,V_D,V_G)= D * exp(V_G / (c_t*kT)) * [1-exp(-|V_D|/kT)] * exp(Phi_b(w)/kT)
                                                                  (Eqs. 1,14)
  dw/dt         = mu_Ron * sign(V_D) * I_D(w,V_D,V_G) * F(w)     (Eq. 16)

FIXED CONSTANTS (taken directly from the paper's Methods, Fig. 2a fit):
  c_t = 83.3, phi_b0 = 0.385 eV, p = 4, T = 300 K (room temperature; not otherwise
  specified in the visible Methods text for the Fig. 2a/4c fits).
  V_th (the MoS2 transistor's ~20V threshold) does NOT appear above: the gate term
  is normalized relative to V_G=0V (exp(V_G/(c_t*kT)), i.e. exp[e(V_G-V_th)/(ctkT)]
  divided by its own value at V_G=0V), since the V_th-dependent part is a pure
  constant offset that is degenerate with the free-fit current scale D anyway --
  this sidesteps needing Fig. 2a's V_th=20V, which belongs to a different device.

FREE-FIT PARAMETERS (fit once, at V_G=0V, against fig4c.csv -- Fig. 4c/4d's device
has different, unreported geometry from the Fig. 2a device the OTHER constants
above were fitted to, so its absolute current/rate scale cannot be reused as-is):
  D       -- current scale.
  mu_Ron  -- rate constant (how fast w moves per unit current per second).
  k_phi   -- barrier-vs-state sensitivity (e/eps_s * sqrt(delta_n/4pi) in the
             paper's notation; delta_n is a doping/state-density product the
             paper itself says cannot be measured independently of w, and Fig.
             4c/4d's own value is unreported, so it must be free here too).
  w0      -- starting internal state (also unconstrained by the paper for this
             device).

SIMPLIFICATIONS (declared explicitly, matching the plan's MVP scope):
  - Only the dominant "image-charge lowering" term of Phi_b (Eq. 8) is kept; the
    second image-charge term (Eq. 9) and the tunnelling term (Eq. 10) are dropped
    -- they introduce additional unconstrained parameters (effective mass m*,
    oxide thickness t_ox, geometrical factors A,B) with no reported values for
    this device, and are described in the paper as OK to neglect at sub-3kT bias
    (main text, discussing Eq. 13).
  - The paper's OCR'd reverse-bias term "1-exp(+e|V_D|/kT)" (Eq. 21) would diverge
    for |V_D| >> kT (e.g. the 30V pulses actually used) -- physically implausible,
    and contradicted by the paper's own statement that "forward-bias current
    saturates quickly with increasing V_D". Read literally this is almost
    certainly an OCR-dropped minus sign; this script uses the SATURATING form
    1-exp(-|V_D|/kT) for both polarities (bounded in [0,1)), and instead encodes
    which direction w moves in via sign(V_D) in the dw/dt equation -- i.e. this
    script's "unified formula" choice replaces the paper's two independently
    fitted branches (different D', phi_b0' etc. per polarity) with ONE branch
    plus an explicit sign.

WHAT THIS SCRIPT DOES NOT DO: change memtransistor.py's runtime behaviour, or
feed back into run_matrix.py's task-accuracy experiments. It is a standalone
device-curve-level comparison (see readme.md's "Physics cross-check" section for
the resulting finding once this has been run).

HONEST LIMITATIONS OF THIS COMPARISON (found on review -- read before trusting
any specific number this script prints):
  1) BOUND-CENSORING: `_fit_phenomenological()` re-fits the CURRENT model's
     dp/dd/kp/kd to each physics-predicted curve, and at extreme V_G that fit
     can land exactly on its own optimizer bounds (device predicted "off" or
     "saturates within one pulse"). A value sitting on a bound means "at least
     this extreme", not a converged point estimate -- ratios computed from such
     values (e.g. "varies Nx across the sweep") are order-of-magnitude
     indicators, not precise figures. `predict_across_vg()` flags this
     explicitly (`at_bound=True`) wherever it happens.
  2) NOT A UNIT-FAIR RMSE COMPARISON BY DEFAULT: this model is fit directly in
     raw current (uA); `fit_device.py`'s reported 0.005/0.012 is computed on
     data NORMALIZED to [0,1]. `fit_to_fig4c()` now reports both the raw-uA and
     the normalized RMSE so the two numbers are comparable on the same footing.
  3) NO GENUINE LTP/LTD ASYMMETRY IS POSSIBLE BY CONSTRUCTION: the "unified
     formula" choice (one D, mu_Ron, k_phi for both polarities, direction set
     only by sign(V_D)) means the model's INSTANTANEOUS response magnitude to
     a +V pulse and a -V pulse at the same w is mathematically identical (see
     `analytic_step_proxy()`, which reports both and shows they match). Any
     *difference* between the re-fit dp and dd numbers above is therefore an
     artifact of where the phenomenological LTP/LTD branch split happens to
     fall on a symmetric underlying trajectory -- NOT a physics prediction of
     asymmetric behaviour. The bound-free, artifact-free way to see "does V_G
     change the per-pulse effect size" is `analytic_step_proxy()`'s output, not
     the dp/dd table.
  4) FIT ROBUSTNESS UNVERIFIED BY A SINGLE START: the V_G=0V fit is a highly
     nonlinear, exponential optimization that can converge to degenerate local
     optima (an earlier version of this script did, silently, until manually
     diagnosed). `fit_to_fig4c()` now retries from several perturbed initial
     guesses and reports the spread across the successful ones, so the reader
     can see whether the reported numbers are stable or fit-dependent.
  5) SOMEWHAT-BUILT-IN CONCLUSION: the model is constructed so V_G multiplies
     current multiplicatively, and current directly sets dw/dt. Given that
     structure, "V_G rescales the per-pulse step" is close to a necessary
     consequence of the setup, not a fully independent discovery -- it should
     be read as "the paper's own equations are AT LEAST consistent with a
     step-size effect, and give no equation-level support for the heuristic's
     window-only, step-flat design", not as a precise, falsifiable prediction.
  6) THE HEADLINE RATIO IS NOT A FIT RESULT -- IT IS exp(dV_G/(c_t*kT)):
     analytic_step_proxy(V_G1)/analytic_step_proxy(V_G2) at the same w and
     |V_D| reduces algebraically to exp((V_G1-V_G2)/(c_t*kT)) -- every fitted
     parameter (D, mu_Ron, k_phi, w0) cancels out of the ratio. This is why the
     multi-start robustness check always returns an IDENTICAL step ratio
     regardless of which local optimum the fit found (confirmed empirically:
     all 6 starts gave exactly the same ratio). So the huge (~10^20x over a
     100V range) number this script reports is really just a restatement of
     the paper's own c_t=83.3 constant -- BORROWED FROM A DIFFERENT DEVICE
     (Fig. 2a, not Fig. 4c/4d) -- extrapolated over the full sweep range, not
     something learned from fitting Fig. 4c/4d's actual data at all. A ratio
     this astronomically large is also not physically plausible for a real
     device (no real transistor's plasticity changes by 20 orders of magnitude
     over 100V; some saturation/leakage/access-resistance floor this
     simplified model omits would dominate long before that). Read the
     headline ratio as "c_t, taken completely at face value with no floor,
     implies an effect far too large to be just a window-size effect" -- a
     qualitative, threshold-like signal (consistent with the paper's own note
     that the device is fully OFF at V_G=-50V) -- not a literal quantitative
     prediction for the Fig. 4c/4d device.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from fit_device import load_fig4c, _sim_ltp, _sim_ltd
from memtransistor import gate_gain

# ---- fixed physical constants (Sangwan 2018, Methods, Fig. 2a fit) ----
C_T = 83.3                 # subthreshold slope factor (dimensionless)
# NOTE: the paper's phi_b0=385meV baseline barrier height is NOT a separate
# constant here -- exp(phi_b0/kT) is degenerate with the free current-scale D
# (see drain_current()'s docstring), so it is folded into D rather than kept
# as its own symbol.
P_WINDOW = 4                 # window-function nonlinearity (paper: "p=4 in Fig. 2a")
K_B = 8.617333262e-5        # Boltzmann constant, eV/K
T_ROOM = 300.0               # room temperature, K (not otherwise specified)
KT = K_B * T_ROOM            # ~0.02585 eV
PULSE_DURATION = 1e-3        # seconds; Fig. 4c/4d used 1 ms pulses (fixed, not fit)


def window_function(w):
    """Eq. 17: F(w) = 1 - [(w-0.5)^2 + 0.75]^p. Zero at w=0,1; max at w=0.5."""
    return 1.0 - ((w - 0.5) ** 2 + 0.75) ** P_WINDOW


def drain_current(w, V_D, V_G, D, k_phi):
    """Unified I_D(w,V_D,V_G) -- Eqs. 1,14, gate term normalized to V_G=0V (see
    module docstring for why V_th cancels out of the ratio).

    Note: PHI_B0 (the ~385meV constant offset in Eq. 2's Phi_b(w) = phi_b0 -
    k_phi*sqrt(w)) does not appear here -- exp(phi_b0/kT) is a pure V_G/V_D/w
    -independent constant, i.e. fully degenerate with the free current-scale D
    (D_eff = D*exp(phi_b0/kT)). Folding it into D directly avoids fitting two
    parameters whose product must nearly cancel a ~10^6 constant, which made the
    optimizer numerically ill-conditioned in an earlier version of this script.
    """
    gate_term = np.exp(np.clip(V_G / (C_T * KT), -700, 700))
    diode_term = 1.0 - np.exp(-np.abs(V_D) / KT)          # always in [0,1); saturating
    barrier_term = np.exp(np.clip(-k_phi * np.sqrt(np.clip(w, 0.0, 1.0)) / KT, -700, 700))
    return D * gate_term * diode_term * barrier_term


def analytic_step_proxy(V_G, params, w=0.5, V_pulse=30.0):
    """Bound-free, artifact-free measure of "how much does one pulse move w":
    the instantaneous |dw/dt|*PULSE_DURATION at a fixed reference state (mid-
    window, w=0.5), for both pulse polarities. Unlike the re-fit dp/dd table,
    this needs no optimizer and cannot hit a bound -- it is a direct evaluation
    of Eq. 16 at one point.

    Returns (step_pos, step_neg): by construction of the "unified formula" (see
    module docstring, limitation 3), these are equal in magnitude -- printing
    both is the honest way to show the model cannot represent genuine LTP/LTD
    asymmetry, rather than silently only ever showing one number.
    """
    D, mu_Ron, k_phi, _w0 = params
    F = window_function(w)
    step_pos = mu_Ron * drain_current(w, +V_pulse, V_G, D, k_phi) * F * PULSE_DURATION
    step_neg = mu_Ron * drain_current(w, -V_pulse, V_G, D, k_phi) * F * PULSE_DURATION
    return step_pos, step_neg


def _dw_dt(t, w, V_D, V_G, D, mu_Ron, k_phi):
    I_D = drain_current(w[0], V_D, V_G, D, k_phi)
    return [mu_Ron * np.sign(V_D) * I_D * window_function(np.clip(w[0], 0.0, 1.0))]


def simulate_pulse_train(V_G, n_ltp, n_ltd, params, V_pulse=30.0):
    """Integrate dw/dt over n_ltp positive-V_pulse pulses followed by n_ltd
    negative-V_pulse pulses (Fig. 4c/4d's protocol), each PULSE_DURATION long.

    Returns:
        I_D_trace: [n_ltp+n_ltd] array, the drain current evaluated at the START
                   of each pulse (before that pulse's update is applied) -- the
                   same convention fit_device.py's _sim_ltp/_sim_ltd use.
    """
    D, mu_Ron, k_phi, w0 = params
    w = float(np.clip(w0, 1e-4, 1 - 1e-4))
    trace = []
    for i in range(n_ltp + n_ltd):
        V_D = V_pulse if i < n_ltp else -V_pulse
        trace.append(drain_current(w, V_D, V_G, D, k_phi))
        sol = solve_ivp(_dw_dt, (0.0, PULSE_DURATION), [w],
                         args=(V_D, V_G, D, mu_Ron, k_phi), method="RK45")
        w = float(np.clip(sol.y[0, -1], 0.0, 1.0))
    return np.array(trace)


def _load_target():
    """Load fig4c.csv as one continuous (pulse_index, current) sequence spanning
    the full LTP-then-LTD protocol, reusing fit_device.py's loader (not
    duplicating it)."""
    ltp, ltd, rng = load_fig4c()
    pulses = np.concatenate([ltp[:, 0], ltd[1:, 0]])     # drop ltd's duplicated anchor point
    current = np.concatenate([ltp[:, 1], ltd[1:, 1]])
    n_ltp, n_ltd = len(ltp), len(ltd) - 1
    return pulses, current, n_ltp, n_ltd, rng


def fit_to_fig4c(verbose=True, n_starts=6, seed=0):
    """Fit (D, mu_Ron, k_phi, w0) at V_G=0V (Fig. 4c/4d's actual measurement
    condition) against fig4c.csv's full LTP+LTD trajectory.

    Tries `n_starts` perturbed initial guesses (this is a highly nonlinear,
    exponential optimization that can converge to degenerate local optima --
    e.g. w rushing to a boundary within the first pulse and freezing there,
    which happened with a naive initial guess during development) and reports
    the spread across the ones that converge, so the caller can see whether
    the reported fit is stable or fit-dependent rather than trusting a single
    run blindly.

    Returns:
        params: (D, mu_Ron, k_phi, w0) of the BEST (lowest-RMSE) start.
        (pulses, current, n_ltp, n_ltd): the target data and protocol geometry.
        rmse_uA: fit quality in raw current units.
        rmse_norm: fit quality normalized to [0,1] like fit_device.py's 0.005/0.012
                   (uses the SAME (i_min,i_max) range fit_device.py normalizes
                   with, for a unit-fair comparison).
        spread: dict with the multi-start diagnostic (rmse_uA min/max/std across
                successful starts, and the same for the downstream
                analytic_step_proxy ratio at V_G=+50 -- the actual number this
                whole exercise cares about).
    """
    pulses, current, n_ltp, n_ltd, rng = _load_target()

    def resid(x):
        sim = simulate_pulse_train(0.0, n_ltp, n_ltd, x)
        return sim - current

    # D's natural scale is directly the current scale (see drain_current()'s
    # docstring -- phi_b0's exp(...) is folded into D), and mu_Ron's natural scale
    # is set so a single pulse moves w by an O(0.05-0.1) fraction of its range:
    # mu_Ron ~ dw / (I_D * F(w) * PULSE_DURATION) ~ 0.075/(D*0.5*1e-3).
    x0_base = np.array([current.max(), 0.075 / (current.max() * 0.5 * PULSE_DURATION), 0.1, 0.5])
    bounds = ([1e-6, 1e-2, -2.0, 0.01], [10.0, 1e8, 2.0, 0.99])

    rng_gen = np.random.default_rng(seed)
    fits = []
    for i in range(n_starts):
        # start 0 = the informed guess; the rest perturb D/mu_Ron log-uniformly
        # and k_phi/w0 uniformly, to probe for other local optima.
        if i == 0:
            x0 = x0_base
        else:
            x0 = np.array([
                x0_base[0] * 10 ** rng_gen.uniform(-1, 1),
                x0_base[1] * 10 ** rng_gen.uniform(-1.5, 1.5),
                rng_gen.uniform(-1.0, 1.0),
                rng_gen.uniform(0.05, 0.95),
            ])
            x0 = np.clip(x0, bounds[0], bounds[1])
        try:
            sol = least_squares(resid, x0, bounds=bounds)
        except Exception:
            continue
        r = resid(sol.x)
        if not np.all(np.isfinite(r)):
            continue
        rmse_uA = float(np.sqrt(np.mean(r ** 2)))
        fits.append((rmse_uA, tuple(sol.x)))

    if not fits:
        raise RuntimeError("physics-model fit: no start converged to a finite residual")
    fits.sort(key=lambda t: t[0])
    rmse_uA, params = fits[0]
    lo, hi = rng
    rmse_norm = rmse_uA / max(hi - lo, 1e-12)

    # Robustness diagnostic: how much does the downstream quantity we actually
    # care about (the +50V/-50V analytic step-size ratio) vary across starts
    # that converged to a reasonable fit (within 3x of the best RMSE)?
    good = [p for r, p in fits if r <= 3 * rmse_uA]
    step_ratios = []
    for p in good:
        s_hi = analytic_step_proxy(50.0, p)[0]
        s_lo = analytic_step_proxy(-50.0, p)[0]
        if s_lo > 0:
            step_ratios.append(s_hi / s_lo)
    spread = dict(n_converged=len(fits), n_good=len(good),
                  rmse_uA_min=fits[0][0], rmse_uA_max=fits[-1][0],
                  step_ratio_min=float(np.min(step_ratios)) if step_ratios else float("nan"),
                  step_ratio_max=float(np.max(step_ratios)) if step_ratios else float("nan"))

    if verbose:
        D, mu_Ron, k_phi, w0 = params
        print(f"Physics-model fit (V_G=0V, best of {len(fits)}/{n_starts} starts): "
              f"D={D:.4g}  mu_Ron={mu_Ron:.4g}  k_phi={k_phi:.4g}  w0={w0:.4g}")
        print(f"  RMSE = {rmse_uA:.4f} uA  ({rmse_norm:.4f} normalized -- compare "
              f"directly to fit_device.py's 0.005 LTP / 0.012 LTD)")
        print(f"  robustness: RMSE across {len(fits)} starts spans "
              f"[{spread['rmse_uA_min']:.4f}, {spread['rmse_uA_max']:.4f}] uA; "
              f"among the {spread['n_good']} within 3x of best, the +50V/-50V step-size "
              f"ratio spans [{spread['step_ratio_min']:.3g}, {spread['step_ratio_max']:.3g}]"
              " (identical across starts is EXPECTED and not a sign of a well-identified fit"
              " -- see docstring limitation 6: this ratio depends only on the fixed c_t"
              " constant, not on the fit)")
    return params, (pulses, current, n_ltp, n_ltd), rmse_uA, rmse_norm, spread


def _normalize01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / max(hi - lo, 1e-12)


# Bounds widened well past fit_device.py's original [1e-3,1.0] (which was tuned
# for gradual, well-behaved real data) so that extreme physics-predicted curves
# (device essentially off, or saturating within one pulse) don't get silently
# clipped to a value that looks like a normal step size -- see module docstring
# limitation 1. `_AT_BOUND_TOL` flags when a fit still lands on an edge anyway.
_STEP_BOUNDS = ([1e-6, 0.0], [5.0, 20.0])
_AT_BOUND_TOL = 1e-3


def _fit_phenomenological(curve_norm, n_ltp, n_ltd):
    """Re-fit the CURRENT phenomenological recursion's dp/dd/kp/kd to a
    physics-predicted (normalized) curve, reusing fit_device.py's own
    _sim_ltp/_sim_ltd (not duplicating them).

    CAVEAT (module docstring limitations 1 and 3): this is an ILLUSTRATIVE,
    secondary view. A dp!=dd result here does NOT mean the physics predicts
    asymmetric LTP/LTD behaviour (the model can't represent that -- see
    analytic_step_proxy()); it reflects where the LTP/LTD branch split falls
    on one symmetric trajectory. And any value at/near a bound is a censored
    estimate, not a converged one (`at_bound` flags this).
    """
    g_ltp = curve_norm[:n_ltp]
    g_ltd = curve_norm[n_ltp - 1:]         # anchor LTD's first point at the LTP peak

    def res_p(x): return _sim_ltp(x[0], x[1], g_ltp[0], len(g_ltp)) - g_ltp
    def res_d(x): return _sim_ltd(x[0], x[1], g_ltd[0], len(g_ltd)) - g_ltd
    sol_p = least_squares(res_p, [0.15, 2.0], bounds=_STEP_BOUNDS)
    sol_d = least_squares(res_d, [0.15, 2.0], bounds=_STEP_BOUNDS)

    lo, hi = _STEP_BOUNDS
    at_bound = any(abs(v - b) < _AT_BOUND_TOL * max(abs(b), 1.0)
                    for v, b in [(sol_p.x[0], lo[0]), (sol_p.x[0], hi[0]),
                                 (sol_d.x[0], lo[0]), (sol_d.x[0], hi[0])])
    return dict(dp=sol_p.x[0], kp=sol_p.x[1], dd=sol_d.x[0], kd=sol_d.x[1], at_bound=at_bound)


def predict_across_vg(params, n_ltp, n_ltd, vgs=(-50, -25, 0, 25, 50)):
    """Hold the fitted params constant; vary only V_G to get physics-PREDICTED
    curves. Reports THREE views, in order of trustworthiness (see module
    docstring):
      - analytic_step: bound-free, artifact-free |dw/dt|*dt at w=0.5 (primary).
      - dp/dd/kp/kd: illustrative re-fit of the CURRENT phenomenological model
        to the predicted curve (secondary; flagged with at_bound).
      - n_states_proxy: pulses spanning the LTP branch's 10-90% range.
    """
    results = {}
    for vg in vgs:
        curve = simulate_pulse_train(float(vg), n_ltp, n_ltd, params)
        curve_norm = _normalize01(curve)
        steps = _fit_phenomenological(curve_norm, n_ltp, n_ltd)
        step_pos, step_neg = analytic_step_proxy(float(vg), params)
        # Effective resolution proxy: pulses needed to span the 10-90% range of
        # the LTP branch (a smaller number = coarser/bigger steps = fewer states).
        g_ltp = curve_norm[:n_ltp]
        span_idx = np.where((g_ltp >= 0.1) & (g_ltp <= 0.9))[0]
        n_states_proxy = float(len(span_idx)) if span_idx.size else float("nan")
        results[vg] = dict(**steps, n_states_proxy=n_states_proxy, curve=curve,
                            analytic_step_pos=step_pos, analytic_step_neg=step_neg)
    return results


def plot_comparison(results, fname="vg_heuristic_vs_physics.png"):
    """Three panels, ordered by trustworthiness (see module docstring):
    (a) PRIMARY -- analytic_step_proxy vs V_G (bound-free, artifact-free) against
        the heuristic's flat (V_G-independent) step-size assumption;
    (b) SECONDARY/illustrative -- the re-fit dp/dd table, with at-bound points
        marked hollow (censored, not converged);
    (c) physics-derived resolution proxy vs V_G against gate_gain(V_G)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vgs = sorted(results.keys())
    step_pos = [results[v]["analytic_step_pos"] for v in vgs]
    dp = [results[v]["dp"] for v in vgs]
    dd = [results[v]["dd"] for v in vgs]
    at_bound = [results[v]["at_bound"] for v in vgs]
    nstates = [results[v]["n_states_proxy"] for v in vgs]
    gate = [gate_gain(v) for v in vgs]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

    ax[0].plot(vgs, step_pos, "o-", color="#4A90D9")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("V_G (V)"); ax[0].set_ylabel("|dw| per pulse at w=0.5 (log scale)")
    ax[0].set_title("PRIMARY: analytic step-size proxy\n(bound-free, symmetric by construction)")

    face = ["white" if b else "#E6613B" for b in at_bound]
    ax[1].plot(vgs, dp, "-", color="#E6613B", alpha=0.4, zorder=1)
    ax[1].scatter(vgs, dp, facecolor=face, edgecolor="#E6613B", zorder=2, label="dp (LTP)")
    face_d = ["white" if b else "#4CB140" for b in at_bound]
    ax[1].plot(vgs, dd, "-", color="#4CB140", alpha=0.4, zorder=1)
    ax[1].scatter(vgs, dd, facecolor=face_d, edgecolor="#4CB140", zorder=2, label="dd (LTD)")
    ax[1].axhline(0.2248, ls="--", color="#E6613B", alpha=0.5, label="heuristic dp (flat)")
    ax[1].axhline(0.8698, ls="--", color="#4CB140", alpha=0.5, label="heuristic dd (flat)")
    ax[1].set_xlabel("V_G (V)"); ax[1].set_ylabel("re-fit step size (normalized)")
    ax[1].set_title("SECONDARY: re-fit dp/dd (illustrative only)\nhollow = at optimizer bound (censored)")
    ax[1].legend(fontsize=7)

    ax[2].plot(vgs, nstates, "s-", color="C3", label="physics-derived resolution proxy")
    ax2b = ax[2].twinx()
    ax2b.plot(vgs, gate, "^--", color="C0", label="heuristic gate_gain(V_G)")
    ax[2].set_xlabel("V_G (V)"); ax[2].set_ylabel("physics: pulses spanning 10-90% (a.u.)")
    ax2b.set_ylabel("heuristic: gate_gain (relative window)")
    ax[2].set_title("Window/resolution vs V_G")
    lines1, labels1 = ax[2].get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax[2].legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")

    fig.tight_layout(); fig.savefig(fname, dpi=130)
    print(f"figure saved: {fname}")


def plot_fit_quality(params, target, fname="vg_physics_fit.png"):
    """Data-vs-model plot at the V_G=0V fit reference, analogous to fit_device.py's
    device_fit.png (fit-quality sanity check)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pulses, current, n_ltp, n_ltd = target
    sim = simulate_pulse_train(0.0, n_ltp, n_ltd, params)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(pulses, current, "o", color="#4A90D9", label="Fig. 4c/4d data (V_G=0V)", markersize=6)
    ax.plot(pulses, sim, "-", color="#4A90D9", label="physics-model fit", linewidth=2)
    ax.axvline(n_ltp - 0.5, ls=":", color="gray", alpha=0.6, label="LTP/LTD polarity switch")
    ax.set_xlabel("Pulse Number"); ax.set_ylabel("Post-synaptic Current (μA)")
    ax.set_title("Physics-model fit vs. Fig. 4c/4d (Sangwan 2018)")
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(fname, dpi=150)
    print(f"figure saved: {fname}")


def run():
    params, target_full, rmse_uA, rmse_norm, spread = fit_to_fig4c()
    pulses, current, n_ltp, n_ltd = target_full
    plot_fit_quality(params, (pulses, current, n_ltp, n_ltd))

    results = predict_across_vg(params, n_ltp, n_ltd)

    print("\nPRIMARY (bound-free, symmetric-by-construction) per-V_G table:")
    print(f"{'V_G':>5} {'|dw| per pulse (w=0.5)':>24} {'gate_gain (heuristic)':>22}")
    for vg in sorted(results.keys()):
        r = results[vg]
        print(f"{vg:>5} {r['analytic_step_pos']:>24.4g} {gate_gain(vg):>22.4f}")

    print("\nSECONDARY/illustrative -- re-fit dp/dd (bound-censored where at_bound=True; "
          "a dp!=dd gap here is a branch-split artifact, not an asymmetry prediction -- "
          "see module docstring limitation 3):")
    print(f"{'V_G':>5} {'dp':>8} {'dd':>8} {'at_bound':>9} {'n_states_proxy':>15}")
    for vg in sorted(results.keys()):
        r = results[vg]
        print(f"{vg:>5} {r['dp']:>8.4f} {r['dd']:>8.4f} {str(r['at_bound']):>9} "
              f"{r['n_states_proxy']:>15.1f}")
    plot_comparison(results)

    step_vals = [results[vg]["analytic_step_pos"] for vg in sorted(results.keys())]
    step_range = max(step_vals) / max(min(step_vals), 1e-300)
    print(f"\nPRIMARY finding: the analytic (bound-free) per-pulse step-size proxy varies "
          f"~{step_range:.2g}x across the V_G sweep (heuristic assumes 1.00x -- flat, by design).")
    print("CAVEAT (docstring limitation 6): this ratio is exp(dV_G/(c_t*kT)) exactly -- it")
    print("depends ONLY on the fixed c_t=83.3 constant borrowed from a DIFFERENT device (Fig.")
    print("2a), not on anything fit to Fig. 4c/4d's own data (every fitted parameter cancels")
    print("out of the ratio -- confirmed by the multi-start check below returning an identical")
    print("ratio regardless of which local optimum was found). Read it as a qualitative,")
    print("threshold-like signal (device off at -50V, saturated by +25V), NOT a literal")
    print("quantitative prediction -- no real device changes by 20 orders of magnitude over")
    print("100V; an unmodeled floor would dominate long before that.")
    print(f"\nFit quality at V_G=0V: RMSE={rmse_uA:.4f} uA / {rmse_norm:.4f} normalized "
          f"(phenomenological fit_device.py, same normalized scale: RMSE 0.005 LTP / 0.012 LTD "
          "-- this model fits noticeably worse, as expected for the single-branch MVP).")
    print(f"Robustness: {spread['n_good']}/{spread['n_converged']} multi-start fits within 3x "
          f"of best RMSE gave a +50V/-50V step ratio in "
          f"[{spread['step_ratio_min']:.3g}, {spread['step_ratio_max']:.3g}] -- identical across "
          "starts, which is itself the evidence for the caveat above, not a sign of a robust fit.")
    print("\nSee this script's module docstring (\"HONEST LIMITATIONS\", esp. #6) before citing "
          "any of the above numbers elsewhere.")


if __name__ == "__main__":
    run()
