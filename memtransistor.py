"""
memtransistor.py
----------------
Phenomenological model of the MoS2 memtransistor (Sangwan 2018, Nature 554:500).
  - Bipolar behavior: Positive pulses trigger LTP, negative pulses trigger LTD (Fig. 4c).
  - Pulse-based: Each update depends on the current state (nonlinear, asymmetric, saturating).
    Applying 'n' pulses corresponds to 'n' sequential state updates, matching the physics of programming.
  - V_G (Gate Voltage): Tunes the usable dynamic range (and thus the number of distinguishable conductance states).
  - Non-idealities can be individually toggled for ablation studies.

State-dependent conductance update step (saturating exponential, matching Fig. 4c characteristics):
  LTP: G <- G + dp * exp(-kp * (G - g_lo)/(g_hi - g_lo))     (large steps near g_lo, small near g_hi)
  LTD: G <- G - dd * exp(-kd * (g_hi - G)/(g_hi - g_lo))     (large steps near g_hi => sudden depression)
The step size decays naturally near the boundaries, capturing the essence of the Sangwan window function.
dp, dd, kp, kd are fitted to Fig. 4c pulse-response data (see fit_device.py).

===== V_G CALIBRATION (Sangwan Fig. 2b/2e) =====
According to the paper: when V_G decreases from +50V to -50V, the switching ratio (I_LRS/I_HRS at VD=0.5V)
drops from 300 down to 8 (~37x). A high switching ratio translates to a wide dynamic range and more
resolvable conductance states. We thus model the number of distinguishable states proportional to the
switching ratio: n_states(V_G) ~ SR(V_G).
The switching ratio (SR) is interpolated in log-space between the two experimental points [(-50V, 8), (+50V, 300)]
(since resistance changes exponentially with gate voltage). gate = SR(V_G)/SR_max in (0, 1] with SR_max = 300 (+50V).
Thus:
  - gate scales the conductance window span; the absolute pulse steps (dp, dd) remain constant.
    Therefore, high V_G (wide span) = many memory states; low V_G = coarse quantization.
  - The W<->G affine mapping does not cancel out the V_G resolution effect because nominal_step does not include the gate factor.

MODELING ASSUMPTIONS (should be declared in the manuscript):
  1) The experimental data of Fig. 4c LTP/LTD is assumed to be measured at the maximum switching ratio regime (gate=1, V_G~+50V);
     the fitted parameters dp/dd are valid at this reference gate voltage. The default V_G is set to this reference (= 50V).
  2) The switching ratio is a multiplicative quantity (I_LRS/I_HRS); we use it as a proxy for the additive conductance window span (phenomenological).
  3) The gate dependence between the limits (-50V and +50V) is modeled via log-interpolation.
"""
from __future__ import annotations
import math
import torch
from device_interface import ConductanceDevice


# ---- Sangwan Fig. 2b/2e: switching ratio (I_LRS/I_HRS) vs V_G ----
_SR_LO, _SR_HI = 8.0, 300.0       # Measured switching ratios at -50V and +50V
_VG_LO, _VG_HI = -50.0, 50.0
_SR_MAX = _SR_HI                   # Normalization reference (gate=1 @ +50V)


def switching_ratio(V_G: float) -> float:
    """V_G -> switching ratio (I_LRS/I_HRS). Interpolates in log-space between the experimental limits."""
    vg = max(_VG_LO, min(_VG_HI, V_G))
    frac = (vg - _VG_LO) / (_VG_HI - _VG_LO)                 # [0, 1]
    return math.exp(math.log(_SR_LO) + frac * (math.log(_SR_HI) - math.log(_SR_LO)))


def gate_gain(V_G: float) -> float:
    """V_G -> usable window / resolution scale factor in (0, 1]. Calibrated to switching ratio: gate = SR(V_G)/SR_max."""
    return min(1.0, switching_ratio(V_G) / _SR_MAX)


class Memtransistor(ConductanceDevice):
    def __init__(self, shape, cfg, torch_device="cpu", dtype=torch.float32, seed=0):
        self.cfg = cfg
        super().__init__(shape, torch_device=torch_device, dtype=dtype, seed=seed)

    # -- Setup & Initialization ------------------------------------------
    def reset(self):
        c = self.cfg
        g = gate_gain(c.V_G)
        # V_G scales the usable window (lower V_G -> smaller switching ratio -> fewer memory states)
        self.g_lo = c.g_min
        self.g_hi = c.g_min + (c.g_max - c.g_min) * g
        self._gate = g

        # Device-to-device variation: static step-size scaling factor assigned at reset
        if c.enable_d2d and c.sigma_d2d > 0:
            noise = torch.randn(self.shape, generator=self.gen, dtype=self.dtype)
            self._d2d = (1.0 + c.sigma_d2d * noise).clamp(min=0.3).to(self.torch_device)
        else:
            self._d2d = torch.ones(self.shape, device=self.torch_device, dtype=self.dtype)

        # Start at the midpoint of the conductance window (corresponds to G_ref for signed weights)
        mid = 0.5 * (self.g_lo + self.g_hi)
        self._G = torch.full(self.shape, mid, device=self.torch_device, dtype=self.dtype)

    def set_state(self, G):
        """Sets the initial conductance G mapped from initial synaptic weights."""
        self._G = G.clamp(self.g_lo, self.g_hi).to(self.torch_device, self.dtype)

    # -- Read Operation ---------------------------------------------------
    def read(self):
        if self.cfg.read_noise > 0:
            n = torch.randn(self.shape, generator=self.gen, dtype=self.dtype).to(self.torch_device)
            return (self._G + self.cfg.read_noise * n).clamp(self.g_lo, self.g_hi)
        return self._G

    # -- Pulse Delta Calculation ------------------------------------------
    def _delta(self, polarity):
        """polarity: +1 for LTP, -1 for LTD, 0 for no update. Returns DeltaG tensor matching shape."""
        c = self.cfg
        span = max(self.g_hi - self.g_lo, 1e-8)
        gnorm = (self._G - self.g_lo) / span          # Normalized conductance G in [0, 1]

        if c.enable_nonlinearity:
            ltp = c.dp * torch.exp(-c.kp * gnorm)         # Large step size near g_lo, decays near g_hi
            ltd = c.dd * torch.exp(-c.kd * (1.0 - gnorm)) # Large step size near g_hi, decays near g_lo (sudden)
        else:
            ltp = torch.full_like(self._G, c.dp)
            ltd = torch.full_like(self._G, c.dd)

        if not c.enable_asymmetry:
            # When asymmetry is disabled, LTD step sizes mirror LTP
            ltd = ltp.clone()

        # Resolution scaling: smaller step_scale = smaller update steps = MORE STATES (quantization ablation)
        ltp = ltp * c.step_scale
        ltd = ltd * c.step_scale

        # NOTE: programming steps are not scaled by the gate factor. The gate only scales the span (window range).
        # Since absolute step size is constant, higher V_G (wider span) leads to more states.
        # W<->G mapping scaling (gamma ~ 1/span) does not cancel this resolution effect.
        ltp = ltp * self._d2d
        ltd = ltd * self._d2d

        step = torch.where(polarity > 0, ltp, torch.where(polarity < 0, -ltd, torch.zeros_like(ltp)))
        return step

    def pulse(self, polarity, n):
        """Applies 'n' sequential programming pulses (vectorized loop up to max(n))."""
        c = self.cfg
        n = n.to(torch.int64)
        max_n = int(n.max().item()) if n.numel() else 0
        for k in range(max_n):
            active = (n > k)                      # Devices that still have pending pulses
            if not active.any():
                break
            step = self._delta(polarity)
            if c.enable_c2c and c.sigma_c2c > 0:
                noise = torch.randn(self.shape, generator=self.gen, dtype=self.dtype).to(self.torch_device)
                step = step + torch.where(step != 0, c.sigma_c2c * noise, torch.zeros_like(step))
            self._G = torch.where(active, (self._G + step).clamp(self.g_lo, self.g_hi), self._G)

    # -- Metadata ---------------------------------------------------------
    @property
    def bounds(self):
        return (self.g_lo, self.g_hi)

    @property
    def nominal_step(self):
        # Nominal average step evaluated at the center of the conductance window.
        # NO GATE FACTOR: actual DeltaG per pulse is gate-independent.
        # Thus w_step = gamma*nominal_step ~ 1/span ~ 1/gate -> V_G controls synaptic resolution.
        return 0.5 * (self.cfg.dp + self.cfg.dd) * self.cfg.step_scale * math.exp(-0.5 * self.cfg.kp)