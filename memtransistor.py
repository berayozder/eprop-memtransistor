"""
memtransistor.py
================
PHENOMENOLOGICAL MODEL of a MoS2 memtransistor (Sangwan 2018, Nature 554:500).

This is the physics of the realistic device -- the heart of what makes the
"memtransistor" arm differ from the perfect "ideal" arm. It is *phenomenological*:
instead of solving the underlying semiconductor equations, it reproduces the device's
measured pulse-response curve (Fig. 4c of the paper) with a small analytic model whose
parameters are FITTED to that real data (see fit_device.py).

KEY BEHAVIOUR
  - Bipolar programming: a positive pulse potentiates (LTP, conductance up), a
    negative pulse depresses (LTD, conductance down).
  - State-dependent, nonlinear, saturating step: how much G moves per pulse depends
    on where G currently is. Near the bottom of the range the up-step is large; near
    the top it shrinks to zero (natural saturation). This is the essence of Sangwan's
    "window function".
  - ASYMMETRY (the core problem): the fitted down-step (dd) is ~4x the up-step (dp),
    so depression is ABRUPT while potentiation is gradual. This abruptness is what
    makes the device's weights coarse (~4 usable levels) and hurts learning.
  - n pulses = n sequential state updates (each pulse sees the state left by the
    previous one), because that is how the real device accumulates.

State-dependent step (saturating exponential, Fig. 4c shape):
  LTP:  G <- G + dp * exp(-kp * (G - g_lo)/(g_hi - g_lo))   (large near g_lo, small near g_hi)
  LTD:  G <- G - dd * exp(-kd * (g_hi - G)/(g_hi - g_lo))   (large near g_hi  => ABRUPT depression)

===== GATE-VOLTAGE (V_G) CALIBRATION (Sangwan Fig. 2b/2e) =====
The gate terminal is the memtransistor's third terminal (PCM devices lack it). The
paper reports that lowering V_G from +50V to -50V drops the switching ratio
(I_LRS/I_HRS) from 300 to 8 (~37x). A higher switching ratio means a wider usable
dynamic range, i.e. more distinguishable conductance states. We therefore make the
number of usable states proportional to the switching ratio:
  - switching_ratio(V_G) is log-interpolated between the two measured points.
  - gate = SR(V_G)/SR_max in (0,1] scales the usable window (span) but NOT the pulse
    step size. So high V_G = wide window + fixed step = MANY states; low V_G = coarse.
  - Because the step is fixed and only the window scales, the weight<->conductance
    mapping (gamma ~ 1/span) does not cancel V_G out -- V_G is a real resolution knob.

MODELLING ASSUMPTIONS (should be declared in the paper):
  1) Fig. 4c/4d were measured at V_G=0V (Sangwan 2018, Fig. 4 caption: "V_G = 0 V
     for all measurements in c, d"); the fitted dp/dd hold at THAT reference, so
     gate_gain(0.0) == 1.0 exactly. V_G != 0 rescales the window relative to that
     reference (up for V_G>0, down for V_G<0) -- it is not capped at 1.0, since
     +50V measurably has ~6x the switching ratio of the 0V reference point.
  2) The switching ratio is a multiplicative quantity (I_LRS/I_HRS); we use it as a
     proxy for an additive number of states -- a phenomenological choice.
  3) The V_G-dependence between the two measured points is a log-interpolation.
"""
from __future__ import annotations
import math
import torch
from device_interface import ConductanceDevice


# ---- Sangwan Fig. 2b/2e: measured switching ratio (I_LRS/I_HRS) vs V_G ----
_SR_LO, _SR_HI = 8.0, 300.0       # switching ratios measured at -50V and +50V
_VG_LO, _VG_HI = -50.0, 50.0      # the two measured gate voltages


def switching_ratio(V_G: float) -> float:
    """Map a gate voltage to the device's switching ratio (I_LRS/I_HRS).

    Log-interpolates between the two measured points (-50V,8) and (+50V,300); clamps
    to those endpoints outside the measured range. Log space is used because device
    resistances vary roughly exponentially with V_G.
    """
    vg = max(_VG_LO, min(_VG_HI, V_G))                       # clamp into measured range
    frac = (vg - _VG_LO) / (_VG_HI - _VG_LO)                 # position in [0,1]
    return math.exp(math.log(_SR_LO) + frac * (math.log(_SR_HI) - math.log(_SR_LO)))


_SR_REF = switching_ratio(0.0)   # ~49.0: normalization reference, gate=1 AT V_G=0V
                                  # (the actual V_G of the Fig. 4c/4d fit -- see Fig. 4 caption)


def gate_gain(V_G: float) -> float:
    """Map V_G to a usable-window / state-count multiplier, relative to the V_G=0V
    reference where the LTP/LTD curve was actually fitted (gate_gain(0.0) == 1.0).

    Calibrated to the switching ratio: gate = SR(V_G)/SR(0V). Since number_of_states
    ~ gate ~ SR, V_G>0V legitimately gives MORE states than the 0V reference (up to
    ~6.1x at +50V) and V_G<0V gives fewer (down to ~0.16x at -50V) -- not capped at
    1.0, unlike the reference point itself which is exactly 1.0 by construction.

    CUSTOM DESIGN, NOT FROM THE PAPER: Sangwan 2018 reports switching ratio (a
    static I-V characteristic, Fig. 2b/2e) and the pulse-response curve (a dynamic
    plasticity characteristic, Fig. 4c/4d) as two SEPARATE measurements on
    (probably) different devices/conditions; the paper never itself connects them.
    Bridging "switching ratio scales the usable window, pulse step stays fixed" is
    this project's own phenomenological choice (see memtransistor_eprop-Sangwan
    ambiguity check in physics_vg_check.py, which finds the paper's own transport
    equations are more consistent with V_G rescaling the STEP instead). No
    literature reference found for this specific window-only bridge -- treat it as
    an assumption, not a cited result (see readme.md "Modeling Assumptions").
    """
    return switching_ratio(V_G) / _SR_REF


class Memtransistor(ConductanceDevice):
    """A parallel array of fitted MoS2 memtransistor cells (one per synaptic weight)."""

    def __init__(self, shape, cfg, torch_device="cpu", dtype=torch.float32, seed=0):
        self.cfg = cfg                         # DeviceConfig with the fitted params + switches
        super().__init__(shape, torch_device=torch_device, dtype=dtype, seed=seed)

    # -- setup ------------------------------------------------------------
    def reset(self):
        """Initialize the conductance window (scaled by V_G), the per-device variation,
        and the starting state (middle of the window)."""
        c = self.cfg
        g = gate_gain(c.V_G)                    # window multiplier from the gate voltage
        # V_G scales the usable window relative to the V_G=0V fit reference (g==1.0):
        # g_max is the window span AT V_G=0V, so g_hi can extend past g_max for V_G>0
        # (more states than the reference) or shrink toward g_min for V_G<0 (fewer).
        self.g_lo = c.g_min                                     # bottom of usable window
        self.g_hi = c.g_min + (c.g_max - c.g_min) * g           # top, scaled by the gate
        self._gate = g                                          # stored for reference

        # device-to-device variation: give each cell its own fixed step multiplier,
        # drawn once at reset (so devices are permanently heterogeneous).
        if c.enable_d2d and c.sigma_d2d > 0:
            noise = torch.randn(self.shape, generator=self.gen, dtype=self.dtype)
            self._d2d = (1.0 + c.sigma_d2d * noise).clamp(min=0.3).to(self.torch_device)
        else:
            self._d2d = torch.ones(self.shape, device=self.torch_device, dtype=self.dtype)

        # Start at the window centre (this is the G_ref for the signed-weight mapping);
        # the synapse layer overwrites this in init_weight to match the initial weights.
        mid = 0.5 * (self.g_lo + self.g_hi)
        self._G = torch.full(self.shape, mid, device=self.torch_device, dtype=self.dtype)

    def set_state(self, G):
        """Directly set the conductance (used by the synapse layer to place initial
        weights). Values are clamped into the usable window."""
        self._G = G.clamp(self.g_lo, self.g_hi).to(self.torch_device, self.dtype)

    # -- read -------------------------------------------------------------
    def read(self):
        """Return the conductance, optionally with additive read noise."""
        if self.cfg.read_noise > 0:
            n = torch.randn(self.shape, generator=self.gen, dtype=self.dtype).to(self.torch_device)
            return (self._G + self.cfg.read_noise * n).clamp(self.g_lo, self.g_hi)
        return self._G

    # -- single-pulse, state-dependent conductance change -----------------
    def _delta(self, polarity):
        """Compute the conductance change DeltaG for ONE pulse, given the current state.

        Args:
            polarity: +1 (LTP), -1 (LTD), or 0 (no pulse), per element.
        Returns:
            step: signed DeltaG per element for a single pulse.
        The step is state-dependent (via gnorm), nonlinear (the exp terms), asymmetric
        (dd != dp), scaled by step_scale and by the per-device d2d multiplier, and
        respects the ablation switches. The gate is deliberately NOT applied to the
        step (only to the window), which is what makes V_G a real resolution knob.

        CUSTOM DESIGN: Sangwan 2018 gives no equation for this (Fig. 4c is presented
        only as measured data + "biexponential fits" in the caption, no formula in
        the text). This specific saturating-exponential-in-gnorm recursion is this
        project's own model, chosen to be fittable to that digitized curve
        (fit_device.py). It is NOT invented in a vacuum, though: nonlinear,
        state-dependent, asymmetric (LTD steeper than LTP) conductance updates vs.
        pulse number are independently well documented for this general class of
        resistive devices, e.g. Gokmen & Vlasov, "Acceleration of Deep Neural
        Network Training with Resistive Cross-Point Devices," Front. Neurosci. 10:333
        (2016) -- that reference supports the general PATTERN (nonlinear + up/down
        asymmetric), not this exact functional form or these fitted constants, which
        remain specific to Sangwan's data.
        """
        c = self.cfg
        span = max(self.g_hi - self.g_lo, 1e-8)
        gnorm = (self._G - self.g_lo) / span          # normalized position in the window, in [0,1]

        if c.enable_nonlinearity:
            ltp = c.dp * torch.exp(-c.kp * gnorm)         # up-step: large near g_lo, small near g_hi
            ltd = c.dd * torch.exp(-c.kd * (1.0 - gnorm)) # down-step: large near g_hi (ABRUPT)
        else:
            ltp = torch.full_like(self._G, c.dp)          # ablation: constant up-step
            ltd = torch.full_like(self._G, c.dd)          # ablation: constant down-step

        if not c.enable_asymmetry:
            ltd = ltp.clone()                             # ablation: make depression == potentiation

        # State-count knob: smaller step_scale => smaller steps => MORE distinguishable states.
        ltp = ltp * c.step_scale
        ltd = ltd * c.step_scale

        # Per-device variation multiplies the step (heterogeneous cells).
        ltp = ltp * self._d2d
        ltd = ltd * self._d2d

        # Choose +ltp for potentiation, -ltd for depression, 0 otherwise.
        step = torch.where(polarity > 0, ltp, torch.where(polarity < 0, -ltd, torch.zeros_like(ltp)))
        return step

    def pulse(self, polarity, n):
        """Apply n pulses per element, each as a sequential state-dependent update.

        We loop max(n) times; on iteration k only the cells that still need pulses
        (n > k) are updated. Each pulse recomputes the state-dependent step and adds
        optional cycle-to-cycle write noise, then clamps to the window. Clamping PER
        PULSE (inside the loop) matches how the real device saturates.
        """
        c = self.cfg
        n = n.to(torch.int64)
        max_n = int(n.max().item()) if n.numel() else 0
        for k in range(max_n):
            active = (n > k)                      # cells that still receive a pulse this round
            if not active.any():
                break
            step = self._delta(polarity)          # state-dependent step for this pulse
            if c.enable_c2c and c.sigma_c2c > 0:  # add cycle-to-cycle write jitter
                noise = torch.randn(self.shape, generator=self.gen, dtype=self.dtype).to(self.torch_device)
                step = step + torch.where(step != 0, c.sigma_c2c * noise, torch.zeros_like(step))
            # Update only active cells; clamp to the usable window (saturation).
            self._G = torch.where(active, (self._G + step).clamp(self.g_lo, self.g_hi), self._G)

    # -- meta -------------------------------------------------------------
    @property
    def bounds(self):
        """(g_lo, g_hi): the usable conductance window (already scaled by V_G)."""
        return (self.g_lo, self.g_hi)

    @property
    def nominal_step(self):
        """Representative average |DeltaG| per pulse at the window centre (gnorm=0.5).

        Used by the writer to convert a desired weight change into a pulse count. It
        mirrors _delta exactly at mid-window: each branch uses its own exponent (kp/kd)
        and it respects the ablation switches. The GATE is intentionally absent, so
        w_step = gamma * nominal_step ~ 1/span ~ 1/gate -- i.e. V_G affects resolution.
        """
        c = self.cfg
        if c.enable_nonlinearity: 
            ltp = c.dp * math.exp(-0.5 * c.kp) 
            ltd = c.dd * math.exp(-0.5 * c.kd)
        else:
            ltp, ltd = c.dp, c.dd
        if not c.enable_asymmetry:
            ltd = ltp
        return 0.5 * (ltp + ltd) * c.step_scale

    @property
    def nominal_ltp_step(self):
        """Pure-LTP average step at the window centre.

        The differential-pair scheme uses ONLY potentiation (never LTD), so its writer
        calibration and its refresh reprogram estimate use this LTP-only step instead
        of the LTP/LTD average above.
        """
        c = self.cfg
        ltp = c.dp * math.exp(-0.5 * c.kp) if c.enable_nonlinearity else c.dp
        return ltp * c.step_scale