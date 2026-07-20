"""
device_interface.py
===================
THE DEVICE CONTRACT (abstraction layer).

This file defines the common interface that every conductance device must provide,
plus the simplest possible device: IdealDevice.

Why this matters: the rest of the system (the synapse writer, the training loop)
only ever talks to this interface -- read(), pulse(), bounds, nominal_step. Because
of that, swapping a perfect IdealDevice for a realistic Memtransistor changes the
results but NOT a single line of the training loop. That clean separation is what
lets us compare "ideal vs realistic device" fairly.

A "device" here is not one physical cell but a whole ARRAY of cells: its state
tensor `_G` has the same shape as the weight matrix it represents, so one device
object stands for thousands of independent memristors (one per synapse).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class ConductanceDevice(ABC):
    """Abstract base class for a population of analog memory cells simulated in parallel.

    Subclasses must implement reset/read/pulse/bounds/nominal_step. The state `_G`
    (conductance) is a tensor whose shape equals the weight matrix's shape, so each
    element is one physical device.
    """

    def __init__(self, shape, torch_device="cpu", dtype=torch.float32, seed=0):
        self.shape = tuple(shape)          # shape of the device array == shape of the weight matrix
        self.torch_device = torch_device   # where tensors live: "cpu" or "mps"
        self.dtype = dtype                 # tensor precision (float32)
        # Dedicated RNG so device noise (c2c/d2d) is reproducible and independent of other randomness.
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self._G = None                     # conductance state tensor; created by reset()
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Initialize/return the conductance state `_G` to its starting configuration."""

    @abstractmethod
    def read(self) -> torch.Tensor:
        """Return the current conductance tensor (possibly with read noise)."""

    @abstractmethod
    def pulse(self, polarity: torch.Tensor, n: torch.Tensor) -> None:
        """Apply programming pulses in place, updating `_G`.

        Args:
            polarity: per-element value in {-1, 0, +1} = depress (LTD) / nothing / potentiate (LTP).
            n:        per-element non-negative integer count of pulses to apply.
        """

    @property
    @abstractmethod
    def bounds(self):
        """Return (g_lo, g_hi): the min and max conductance the device can hold."""

    @property
    @abstractmethod
    def nominal_step(self) -> float:
        """A representative average LTP step size, used by the writer to convert a
        desired weight change into an integer number of pulses."""

    @property
    def G(self):
        """Convenience read-only accessor for the raw state tensor."""
        return self._G


class IdealDevice(ConductanceDevice):
    """A perfect, linear, unbounded, noiseless memory -- the theoretical ceiling.

    This is NOT a real device (no memtransistor, no PCM). It represents "what if the
    device imposed no limits at all?" Its conductance equals the weight directly
    (W = G), it has infinite range, and every pulse moves it by exactly nominal_step
    with no state-dependence and no noise. The gap between this arm and the realistic
    memtransistor arm is precisely the cost of the device's physical flaws.
    """

    def reset(self):
        # Start every weight at 0 (an ideal weight can take any real value).
        self._G = torch.zeros(self.shape, device=self.torch_device, dtype=self.dtype)

    def read(self):
        # No read noise, no drift: return the exact stored value.
        return self._G

    def pulse(self, polarity, n):
        # Each pulse moves G by exactly nominal_step (=1.0); perfectly linear, no clamping.
        # (In practice the ideal arm updates its weight continuously via the synapse writer,
        # so this pulse path is rarely used, but it keeps the interface complete.)
        self._G = self._G + polarity * n.to(self.dtype) * self.nominal_step

    @property
    def bounds(self):
        # Infinite range: an ideal weight is never clamped. The synapse layer detects
        # these infinite bounds and switches to a direct W = G mapping.
        return (-float("inf"), float("inf"))

    @property
    def nominal_step(self):
        # Unit step: 1 pulse == 1.0 change in G (== 1.0 change in weight for the ideal map).
        return 1.0