"""
memtransistor.py
----------------
PHENOMENOLOGICAL model of the MoS2 memtransistor (Sangwan 2018, Nature 554:500).
  - Bipolar: positive pulse LTP, negative pulse LTD (Fig. 4c).
  - Pulse-based: each pulse applies a state-dependent, nonlinear, asymmetric, saturating
    DeltaG. n pulses = n consecutive state updates (this is the physical behavior).
  - V_G: sets the available dynamic range (=> number of distinguishable states).
  - Non-idealities can be turned off individually for ablation studies.

State-dependent step (saturating exponential, Fig. 4c shape):
  LTP:  G <- G + dp * exp(-kp * (G - g_lo)/(g_hi - g_lo))     (large near g_lo, small near g_hi)
  LTD:  G <- G - dd * exp(-kd * (g_hi - G)/(g_hi - g_lo))     (large near g_hi => ABRUPT depression)
Step naturally decays near the boundary -> essence of Sangwan window function (eq.17).
dp, dd, kp, kd are fitted to Fig. 4c pulse-response (see fit_device.py).

===== V_G CALIBRATION (Sangwan Fig. 2b/2e) =====
Paper: As V_G sweeps from +50V down to -50V, the switching ratio (I_LRS/I_HRS, VD=0.5V)
DROPS from 300 to 8 (~37x). High switching ratio = wide dynamic range =
more distinguishable conductance states. Therefore we take the NUMBER OF
DISTINGUISHABLE STATES to be proportional to the switching ratio: n_states(V_G) ~ SR(V_G).
SR is log-interpolated between the two measured points [(-50V,8), (+50V,300)]
(because resistances change exponentially with V_G). gate = SR(V_G)/SR_max in (0,1],
SR_max=300 (+50V). Thus:
  - gate scales the window (span); pulse steps (dp, dd) remain ABSOLUTE constants
    -> high V_G = wide window + constant step = MANY STATES; low V_G = coarse.
  - W<->G affine mapping DOES NOT cancel out V_G (gate is not in nominal_step; see below).

MODELING ASSUMPTIONS (should be declared in the paper):
  1) Fig. 4c LTP/LTD is assumed to be measured at MAX switching ratio regime
     (gate=1, V_G~+50V); fitted dp/dd apply to this reference. (The paper does not
     explicitly state the V_G for Fig. 4c.) Default V_G is fixed to this reference (=50).
  2) switching ratio is a MULTIPLICATIVE quantity (I_LRS/I_HRS); we use it as a proxy
     for the number of distinguishable states (ADDITIVE window) -> phenomenological.
  3) exact V_G-dependence of gate (outside 300<->8) is a log-interpolation.
"""
from __future__ import annotations
import math
import torch
from device_interface import ConductanceDevice


# ---- Sangwan Fig. 2b/2e: switching ratio (I_LRS/I_HRS) vs V_G ----
_SR_LO, _SR_HI = 8.0, 300.0       # measured switching ratios at -50V and +50V
_VG_LO, _VG_HI = -50.0, 50.0
_SR_MAX = _SR_HI                   # normalization reference (gate=1 @ +50V)


def switching_ratio(V_G: float) -> float:
    """V_G -> switching ratio (I_LRS/I_HRS). Log-interpolation between
    measured [(-50,8), (+50,300)]; clipped to extreme values outside the range."""
    vg = max(_VG_LO, min(_VG_HI, V_G))
    frac = (vg - _VG_LO) / (_VG_HI - _VG_LO)                 # [0,1]
    return math.exp(math.log(_SR_LO) + frac * (math.log(_SR_HI) - math.log(_SR_LO)))


def gate_gain(V_G: float) -> float:
    """V_G -> usable window / distinguishable state multiplier (0,1].
    Calibrated to Sangwan switching ratio: gate = SR(V_G)/SR_max.
    Since n_states ~ gate ~ SR, V_G effect reflects the measured ~37x range."""
    return min(1.0, switching_ratio(V_G) / _SR_MAX)


class Memtransistor(ConductanceDevice):
    def __init__(self, shape, cfg, torch_device="cpu", dtype=torch.float32, seed=0):
        self.cfg = cfg
        super().__init__(shape, torch_device=torch_device, dtype=dtype, seed=seed)

    # -- setup ------------------------------------------------------------
    def reset(self):
        c = self.cfg
        g = gate_gain(c.V_G)
        # V_G scales the usable window (low V_G -> small switching ratio -> few states)
        self.g_lo = c.g_min
        self.g_hi = c.g_min + (c.g_max - c.g_min) * g
        self._gate = g

        # device-to-device: a fixed step-multiplier for each device (drawn at reset)
        if c.enable_d2d and c.sigma_d2d > 0:
            noise = torch.randn(self.shape, generator=self.gen, dtype=self.dtype)
            self._d2d = (1.0 + c.sigma_d2d * noise).clamp(min=0.3).to(self.torch_device)
        else:
            self._d2d = torch.ones(self.shape, device=self.torch_device, dtype=self.dtype)

        # start from midpoint (G_ref for signed weights); Synapse init can overwrite this
        mid = 0.5 * (self.g_lo + self.g_hi)
        self._G = torch.full(self.shape, mid, device=self.torch_device, dtype=self.dtype)

    def set_state(self, G):
        """Used by Synapse to map the initial weight to G."""
        self._G = G.clamp(self.g_lo, self.g_hi).to(self.torch_device, self.dtype)

    # -- read -------------------------------------------------------------
    def read(self):
        if self.cfg.read_noise > 0:
            n = torch.randn(self.shape, generator=self.gen, dtype=self.dtype).to(self.torch_device)
            return (self._G + self.cfg.read_noise * n).clamp(self.g_lo, self.g_hi)
        return self._G

    # -- state-dependent DeltaG for a single pulse ------------------------
    def _delta(self, polarity):
        """polarity: +1 LTP, -1 LTD, 0 none. Shape == shape. DeltaG for one pulse."""
        c = self.cfg
        span = max(self.g_hi - self.g_lo, 1e-8)
        gnorm = (self._G - self.g_lo) / span          # [0,1]

        if c.enable_nonlinearity:
            ltp = c.dp * torch.exp(-c.kp * gnorm)         # large at g_lo, small at g_hi
            ltd = c.dd * torch.exp(-c.kd * (1.0 - gnorm)) # large at g_hi, small at g_lo (ABRUPT)
        else:
            ltp = torch.full_like(self._G, c.dp)
            ltd = torch.full_like(self._G, c.dd)

        if not c.enable_asymmetry:
            # turn off asymmetry: make LTD step identical to LTP
            ltd = ltp.clone()

        # number of states knob: small step_scale = small step = MANY STATES (quantization ablation)
        ltp = ltp * c.step_scale
        ltd = ltd * c.step_scale

        # NOTE: GATE IS NOT APPLIED TO the pulse step. Gate only scales the window (span);
        # absolute step is constant -> high V_G (wide span) = more states. Thus
        # W<->G affine mapping (gamma ~ 1/span) does not cancel out V_G.
        ltp = ltp * self._d2d
        ltd = ltd * self._d2d

        step = torch.where(polarity > 0, ltp, torch.where(polarity < 0, -ltd, torch.zeros_like(ltp)))
        return step

    def pulse(self, polarity, n):
        """n pulses = n consecutive state-dependent updates (vectorized, loops max(n) times)."""
        c = self.cfg
        n = n.to(torch.int64)
        max_n = int(n.max().item()) if n.numel() else 0
        for k in range(max_n):
            active = (n > k)                      # devices that still receive pulses in this round
            if not active.any():
                break
            step = self._delta(polarity)
            if c.enable_c2c and c.sigma_c2c > 0:
                noise = torch.randn(self.shape, generator=self.gen, dtype=self.dtype).to(self.torch_device)
                step = step + torch.where(step != 0, c.sigma_c2c * noise, torch.zeros_like(step))
            self._G = torch.where(active, (self._G + step).clamp(self.g_lo, self.g_hi), self._G)

    # -- meta -------------------------------------------------------------
    @property
    def bounds(self):
        return (self.g_lo, self.g_hi)

    @property
    def nominal_step(self):
        # writer scaling: average |DeltaG|/pulse at the middle of the window (gnorm=0.5).
        # EXACTLY the same logic as _delta: each branch uses its own exponential (kp/kd) and
        # obeys ablation flags. NO GATE (gate widens span, not step) ->
        # w_step = gamma*nominal_step ~ 1/span ~ 1/gate, meaning V_G affects resolution.
        c = self.cfg
        if c.enable_nonlinearity:
            ltp = c.dp * math.exp(-0.5 * c.kp)
            ltd = c.dd * math.exp(-0.5 * c.kd)
        else:
            ltp, ltd = c.dp, c.dd
        if not c.enable_asymmetry:
            ltd = ltp
        return 0.5 * (ltp + ltd) * c.step_scale