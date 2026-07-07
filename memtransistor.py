"""
memtransistor.py
----------------
Phenomenological model of the MoS2 memtransistor (Sangwan 2018).
  - Bipolar behavior: Positive pulses trigger LTP, negative pulses trigger LTD (Fig. 4c).
  - Pulse-based: Each update depends on the current state (nonlinear, asymmetric, saturating).
    Applying 'n' pulses corresponds to 'n' sequential state updates, matching the physics of programming.
  - V_G (Gate Voltage): Controls the dynamic range and update granularity (Fig. 2b inset:
    switching ratio varies with V_G from ~300 to ~8). It scales the usable conductance window and step sizes.
  - Non-idealities can be individually toggled for ablation studies.

State-dependent conductance update step (saturating exponential, matching Fig. 4c characteristics):
  LTP: G <- G + dp * exp(-kp * (G - g_lo)/(g_hi - g_lo))     (large steps near g_lo, small near g_hi)
  LTD: G <- G - dd * exp(-kd * (g_hi - G)/(g_hi - g_lo))     (large steps near g_hi, small near g_lo)
The step size decays naturally near the boundaries, capturing the essence of the Sangwan window function.
"""
from __future__ import annotations
import math
import torch
from device_interface import ConductanceDevice


def gate_gain(V_G: float) -> float:
    """
    Computes V_G -> usable dynamic range / granularity scaling factor in (0, 1].
    High V_G -> wider conductance window, more resolvable memory states.
    V_G range ~ [-50, 50] maps to ~ [0.15, 1.0], matching the switching ratio trend in Fig. 2b.
    """
    return 0.15 + 0.85 * (1.0 / (1.0 + math.exp(-0.08 * V_G)))


class Memtransistor(ConductanceDevice):
    def __init__(self, shape, cfg, torch_device="cpu", dtype=torch.float32, seed=0):
        self.cfg = cfg
        super().__init__(shape, torch_device=torch_device, dtype=dtype, seed=seed)

    # -- Setup & Initialization ------------------------------------------
    def reset(self):
        c = self.cfg
        g = gate_gain(c.V_G)
        # V_G restricts the dynamic range (low V_G yields small switching ratio)
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
        """
        Calculates the single-pulse DeltaG update step.
        polarity: +1 for LTP, -1 for LTD, 0 for no update.
        Shape matches self.shape.
        """
        c = self.cfg
        span = max(self.g_hi - self.g_lo, 1e-8)
        gnorm = (self._G - self.g_lo) / span          # Normalized conductance in [0,1]

        if c.enable_nonlinearity:
            ltp = c.dp * torch.exp(-c.kp * gnorm)         # Large step size near g_lo, decays near g_hi
            ltd = c.dd * torch.exp(-c.kd * (1.0 - gnorm)) # Large step size near g_hi, decays near g_lo
        else:
            ltp = torch.full_like(self._G, c.dp)
            ltd = torch.full_like(self._G, c.dd)

        if not c.enable_asymmetry:
            # When asymmetry is disabled, LTD step sizes mirror LTP
            ltd = ltp.clone()

        # Gate voltage granularity scale: lower V_G leads to smaller conductance windows & step sizes
        ltp = ltp * self._gate * self._d2d
        ltd = ltd * self._gate * self._d2d

        step = torch.where(polarity > 0, ltp, torch.where(polarity < 0, -ltd, torch.zeros_like(ltp)))
        return step

    def pulse(self, polarity, n):
        """Applies 'n' pulses sequentially to the device population (iteratively updated 'n' times)."""
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
        # Nominal average LTP step size evaluated at the center of the conductance window
        return 0.5 * (self.cfg.dp + self.cfg.dd) * self._gate * math.exp(-0.5 * self.cfg.kp)