"""
synapse.py
==========
THE BRIDGE between the network's world (signed weights W) and the device's world
(non-negative, bounded conductances G), plus the WRITER that turns a desired weight
change into physical pulses.

Two mappings are supported:

  SINGLE bipolar device (default):
      W = gamma * (G - G_ref),   G_ref = middle of the window,  gamma = w_range / (half window)
      One device per weight. To increase W you send LTP pulses; to decrease W you send
      LTD pulses. This is where the device's abrupt LTD hurts the most.

  DIFFERENTIAL PAIR (mitigation, differential_pair=True):
      W = gamma * (G_pos - G_neg),  G_ref = g_lo,  gamma = w_range / (full window)
      Two devices per weight. BOTH are programmed with LTP-only (upward) pulses:
        - increase W  -> potentiate G_pos
        - decrease W  -> potentiate G_neg
      Because it never uses the abrupt LTD, the differential pair sidesteps the core
      problem. Its cost: two devices per weight, plus a periodic "refresh" when both
      devices creep up together.

      CUSTOM DESIGN, but the general ARCHITECTURE (not our specific refresh logic)
      is an established neuromorphic-hardware technique: representing one weight as
      the difference of two device conductances is described e.g. in Nair, Muller &
      Indiveri, "A differential memristive synapse circuit for on-line learning in
      neuromorphic computing systems," arXiv:1709.05484 (2017); saturation-driven
      refresh of such pairs is likewise a recognized general concept in that
      literature (e.g. US Patent 10,445,640, "Scalable refresh for asymmetric
      non-volatile memory-based neuromorphic circuits"). This project's SPECIFIC
      refresh trigger (min(G_pos,G_neg) >= threshold, see refresh_saturated() below)
      and its energy-accounting method are our own implementation, informed by but
      not copied from either source (we could not verify their exact trigger
      condition against the primary text within this review).

The WRITER converts the ideal weight change requested by e-prop into an integer number
of pulses (weights are continuous; the device moves in discrete steps). Three schemes:
  - "direct":     round(dW / w_step) pulses, applied immediately.
  - "accumulate": keep a high-precision residual, emit pulses only when it exceeds one
                  step (this is the mixed-precision / gradient-accumulation idea; default).
  - "verify":     read-modify-write loop until the target weight is reached.

CUSTOM DESIGN: none of these three schemes come from either paper (Bellec 2020 is
silent on hardware writers; Sangwan 2018 doesn't address multi-pulse programming
strategy). They follow standard, generic non-volatile-memory engineering patterns
("accumulate" = mixed-precision/stochastic-rounding-style residual carry, as used
e.g. in Gokmen & Vlasov 2016; "verify" = program-verify / ISPP, textbook-standard in
flash/RRAM programming) rather than being specific to a single citable source.
"""
from __future__ import annotations
import torch


class Synapse:
    """Owns the device(s) for one weight matrix and handles W<->G mapping + writing."""

    def __init__(self, device, w_range, writer="accumulate", verify_max_iter=5,
                 differential_pair=False, auto_refresh=True):
        """
        Args:
            device: either a single ConductanceDevice, or a tuple (dev_pos, dev_neg)
                    for a differential pair.
            w_range: weights are mapped to [-w_range, +w_range].
            writer: "direct" | "accumulate" | "verify".
            verify_max_iter: iteration cap for the "verify" writer.
            differential_pair: flag (only takes effect if a device pair is provided).
            auto_refresh: whether to refresh saturated differential pairs automatically.
        """
        # Accept either a single device or a (positive, negative) device pair.
        if isinstance(device, tuple):
            self.dev_pos, self.dev_neg = device
            self.dev = self.dev_pos                 # `self.dev` is the reference device for meta/shape
            self.differential_pair = True
        else:
            self.dev_pos = device
            self.dev_neg = None
            self.dev = device
            # A differential pair REQUIRES two devices; with a single device it stays off.
            self.differential_pair = differential_pair and (self.dev_neg is not None)

        self.auto_refresh = auto_refresh
        g_lo, g_hi = self.dev_pos.bounds

        # Ideal devices report infinite bounds -> use a direct identity map (W = G).
        if not (g_lo > -1e30 and g_hi < 1e30):
            self.gamma = 1.0                        # weight-per-conductance scale
            self.g_ref = 0.0                        # reference conductance (zero for ideal)
            self.ideal_map = True
        else:
            # differential: reference at g_lo, full window; single: reference at centre, half window.
            self.g_ref = g_lo if self.differential_pair else 0.5 * (g_lo + g_hi)
            span = (g_hi - g_lo) if self.differential_pair else 0.5 * (g_hi - g_lo)
            self.gamma = w_range / max(span, 1e-8)
            self.ideal_map = False

        self.writer = writer
        self.verify_max_iter = verify_max_iter
        # w_step = weight change delivered by ONE nominal pulse. The differential writer
        # uses the LTP-only step (it never depresses); everyone else uses the LTP/LTD average.
        if self.differential_pair and hasattr(self.dev, "nominal_ltp_step"):
            self.w_step = self.gamma * self.dev.nominal_ltp_step
        else:
            self.w_step = self.gamma * self.dev.nominal_step
        # acc = high-precision residual buffer for the "accumulate" writer (per weight).
        self.acc = torch.zeros(self.dev.shape, device=self.dev.torch_device, dtype=self.dev.dtype)
        self.n_pulses_total = 0                     # running count of pulses (an energy proxy)

    # -- weight <-> conductance mapping -----------------------------------
    def weight(self):
        """Read the current weights W from the device conductance(s)."""
        if self.differential_pair:
            return self.gamma * (self.dev_pos.read() - self.dev_neg.read())
        return self.gamma * (self.dev.read() - self.g_ref)

    def init_weight(self, W):
        """Place initial weights W onto the device conductance state(s)."""
        if self.differential_pair:
            # Positive weight -> difference sits on G_pos; negative -> on G_neg; other device at g_lo.
            G_pos = torch.where(W >= 0, self.g_ref + W / self.gamma, torch.full_like(W, self.g_ref))
            G_neg = torch.where(W < 0, self.g_ref - W / self.gamma, torch.full_like(W, self.g_ref))
            self.dev_pos.set_state(G_pos)
            self.dev_neg.set_state(G_neg)
            return
        if self.ideal_map:
            self.dev._G = W.clone().to(self.dev.torch_device, self.dev.dtype)
        else:
            G = self.g_ref + W / self.gamma
            if hasattr(self.dev, "set_state"):
                self.dev.set_state(G)
            else:
                self.dev._G = G

    # -- differential-pair saturation refresh -----------------------------
    def refresh_saturated(self, threshold_ratio=0.92):
        """CUSTOM DESIGN (see module docstring): the differential-pair architecture
        is a known technique (Nair, Muller & Indiveri 2017); this specific
        min(G_pos,G_neg)-based trigger, the 0.92 threshold, and the pulse-based
        energy accounting below are this project's own, not taken from a paper.

        Remove the common-mode offset from differential pairs that have crept up.

        Over time, LTP-only updates push BOTH devices of a pair upward. When both are
        high, the pair loses headroom. Refresh subtracts the shared offset: it resets
        both devices toward g_lo while preserving the difference W = gamma*(G_pos-G_neg).

        CORRECT TRIGGER: min(G_pos, G_neg) >= threshold (both devices high = a removable
        common offset exists). A large legitimate weight keeps ONE device near g_hi; that
        one-sided saturation is NOT removable, so it must not trigger refresh (otherwise
        it would fire every trial and waste pulses -- the "churn" bug we fixed).

        The physical cost (2 RESET pulses per pair + LTP pulses to rebuild the difference)
        is counted into n_pulses_total for a fair energy comparison.
        """
        if not self.differential_pair:
            return
        g_lo, g_hi = self.dev_pos.bounds
        sat_th = g_lo + threshold_ratio * (g_hi - g_lo)
        sat = torch.minimum(self.dev_pos._G, self.dev_neg._G) >= sat_th   # both devices high
        if not sat.any():
            return
        W_curr = self.weight()
        g_diff = W_curr / self.gamma                                       # = G_pos - G_neg (preserved)
        # Rebuild the minimal representation: put the whole difference on one device, other at g_lo.
        G_pos_new = torch.where(sat, torch.where(g_diff >= 0, g_lo + g_diff, torch.full_like(g_diff, g_lo)), self.dev_pos._G)
        G_neg_new = torch.where(sat, torch.where(g_diff < 0, g_lo - g_diff, torch.full_like(g_diff, g_lo)), self.dev_neg._G)

        # Count the refresh energy: 2 RESET pulses per saturated pair + LTP pulses to
        # rebuild the difference from g_lo (estimated with the LTP-only step).
        n_sat = int(sat.sum().item())
        step = self.dev_pos.nominal_ltp_step if hasattr(self.dev_pos, "nominal_ltp_step") else self.dev_pos.nominal_step
        reprogram = int(((G_pos_new[sat] - g_lo).sum() + (G_neg_new[sat] - g_lo).sum()).item() / max(step, 1e-9))
        self.n_pulses_total += (2 * n_sat + reprogram)

        self.dev_pos._G = G_pos_new.clamp(g_lo, g_hi)
        self.dev_neg._G = G_neg_new.clamp(g_lo, g_hi)

    # -- writing ----------------------------------------------------------
    def _emit(self, dW_pulses, max_pulses=30):
        """Apply a signed integer pulse-count tensor to the device(s).

        Args:
            dW_pulses: per-weight signed pulse count (sign = direction, magnitude = count).
            max_pulses: per-update budget clamp (prevents huge single-step writes).
        """
        dW_pulses = torch.clamp(dW_pulses, -max_pulses, max_pulses)
        if self.differential_pair:
            # LTP-only routing: positive dW -> potentiate G_pos; negative dW -> potentiate G_neg.
            n_pos = torch.relu(dW_pulses)               # positive part goes to the + device
            n_neg = torch.relu(-dW_pulses)              # negative part (as a positive count) to the - device
            self.dev_pos.pulse(torch.ones_like(n_pos), n_pos)   # polarity +1 = LTP for both
            self.dev_neg.pulse(torch.ones_like(n_neg), n_neg)
            self.n_pulses_total += int(n_pos.sum().item() + n_neg.sum().item())
            if self.auto_refresh:
                self.refresh_saturated()
            return
        # Single-device path: sign = polarity (LTP/LTD), magnitude = number of pulses.
        polarity = torch.sign(dW_pulses)
        n = dW_pulses.abs()
        self.dev.pulse(polarity, n)
        self.n_pulses_total += int(n.sum().item())

    def update(self, desired_dW):
        """Write a desired weight change onto the device using the configured writer.

        Args:
            desired_dW: per-weight continuous change requested by the learning rule.
        """
        # Ideal device: apply the continuous change exactly (no pulses, the theoretical ceiling).
        if self.ideal_map:
            self.dev._G = self.dev._G + desired_dW
            return

        if self.writer == "direct":
            # Round the desired change to the nearest whole number of steps.
            n = torch.round(desired_dW / self.w_step)
            self._emit(n)

        elif self.writer == "accumulate":
            # Mixed-precision: accumulate the desired change, emit only whole pulses,
            # and keep the sub-step remainder for next time (nothing is lost).
            self.acc = self.acc + desired_dW
            n = torch.trunc(self.acc / self.w_step)      # only complete pulses
            self._emit(n)
            self.acc = self.acc - n * self.w_step        # retain the residual

        elif self.writer == "verify":
            # Closed loop: read back, correct, repeat until the target weight is reached.
            target_W = self.weight() + desired_dW
            for _ in range(self.verify_max_iter):
                err = target_W - self.weight()
                n = torch.round(err / self.w_step)
                if (n == 0).all():
                    break
                self._emit(n)
        else:
            raise ValueError(self.writer)