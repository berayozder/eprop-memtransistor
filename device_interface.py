"""
device_interface.py
-------------------
Device contract interface. The Synapse/Writer only sees this abstraction.
The only difference between IdealDevice and Memtransistor is the device object instance,
while the surrounding training loop code remains completely identical.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class ConductanceDevice(ABC):
    """Abstract class for a population of analog memory devices simulated in parallel (shape matches weight tensor shape)."""

    def __init__(self, shape, torch_device="cpu", dtype=torch.float32, seed=0):
        self.shape = tuple(shape)
        self.torch_device = torch_device
        self.dtype = dtype
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self._G = None
        self.reset()

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def read(self) -> torch.Tensor: ...

    @abstractmethod
    def pulse(self, polarity: torch.Tensor, n: torch.Tensor) -> None:
        """
        Applies programming pulses to W.
        polarity: sign of pulse in {-1, 0, +1} (LTD / none / LTP)
        n: tensor of pulse counts (n >= 0, integer)
        Updates internal state G in-place.
        """

    @property
    @abstractmethod
    def bounds(self): ...

    @property
    @abstractmethod
    def nominal_step(self) -> float:
        """Nominal average LTP step size used by the Writer for scaling pulse counts to conductance updates."""

    @property
    def G(self):
        return self._G


class IdealDevice(ConductanceDevice):
    """Linear, unbounded, and noiseless reference device. Represents the baseline / theoretical ceiling."""

    def reset(self):
        self._G = torch.zeros(self.shape, device=self.torch_device, dtype=self.dtype)

    def read(self):
        return self._G

    def pulse(self, polarity, n):
        # Every programming pulse applies exactly the nominal step; no non-idealities or limits
        self._G = self._G + polarity * n.to(self.dtype) * self.nominal_step

    @property
    def bounds(self):
        return (-float("inf"), float("inf"))

    @property
    def nominal_step(self):
        return 1.0