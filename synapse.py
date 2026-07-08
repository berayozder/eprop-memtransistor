"""
synapse.py
----------
Synaptic weight-to-conductance mapping bridge + PULSE programming writer.

W <-> G mapping (using a single bipolar memristive device):
    W = gamma * (G - G_ref),   G_ref = center of the conductance window
    gamma = w_range / (half conductance window)
Because MoS2 memtransistors are bipolar, signed weights can be mapped to a single device.

Programming Writer Schemes:
    - direct: Instantly converts target DeltaW to pulses and writes to device.
    - accumulate: Accumulates micro-updates in a high-precision digital buffer;
      applies physical programming pulses only when accumulated error exceeds 1 step (e.g. Demirag et al., 2021).
    - verify: Closed-loop read-write scheme; iteratively applies single pulses until the weight matches the target.
"""
from __future__ import annotations
import torch


class Synapse:
    def __init__(self, device, w_range, writer="accumulate", verify_max_iter=5):
        self.dev = device
        g_lo, g_hi = device.bounds
        # Ideal devices have infinite boundaries -> direct mapping (W = G)
        if not (g_lo > -1e30 and g_hi < 1e30):
            self.gamma = 1.0
            self.g_ref = 0.0
            self.ideal_map = True
        else:
            self.g_ref = 0.5 * (g_lo + g_hi)
            self.gamma = w_range / max(0.5 * (g_hi - g_lo), 1e-8)
            self.ideal_map = False
        self.writer = writer
        self.verify_max_iter = verify_max_iter
        self.w_step = self.gamma * device.nominal_step   # synaptic weight change corresponding to 1 pulse
        self.acc = torch.zeros(device.shape, device=device.torch_device, dtype=device.dtype)
        self.n_pulses_total = 0                          # Total applied pulse counter (energy/overhead proxy)

    # -- Weight Mapping ---------------------------------------------------
    def weight(self):
        return self.gamma * (self.dev.read() - self.g_ref)

    def init_weight(self, W):
        """Maps initial weights to physical conductance states."""
        if self.ideal_map:
            self.dev._G = W.clone().to(self.dev.torch_device, self.dev.dtype)
        else:
            G = self.g_ref + W / self.gamma
            if hasattr(self.dev, "set_state"):
                self.dev.set_state(G)
            else:
                self.dev._G = G

    # -- Writing / Programming --------------------------------------------
    def _emit(self, dW_pulses, max_pulses=30):
        """Applies signed pulse counts to the physical device population."""
        dW_pulses = torch.clamp(dW_pulses, -max_pulses, max_pulses)
        polarity = torch.sign(dW_pulses)
        n = dW_pulses.abs()
        self.dev.pulse(polarity, n)
        self.n_pulses_total += int(n.sum().item())

    def update(self, desired_dW):
        # Ideal device: Continuous, direct W update (theoretical ceilings)
        if self.ideal_map:
            self.dev._G = self.dev._G + desired_dW
            return

        if self.writer == "direct":
            n = torch.round(desired_dW / self.w_step)
            self._emit(n)

        elif self.writer == "accumulate":
            self.acc = self.acc + desired_dW
            n = torch.trunc(self.acc / self.w_step)        # extract whole pulse counts
            self._emit(n)
            self.acc = self.acc - n * self.w_step          # retain fractional remainders

        elif self.writer == "verify":
            target_W = self.weight() + desired_dW
            for _ in range(self.verify_max_iter):
                err = target_W - self.weight()
                n = torch.round(err / self.w_step)
                if (n == 0).all():
                    break
                self._emit(n)
        else:
            raise ValueError(self.writer)