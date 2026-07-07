"""
device_interface.py
-------------------
Cihaz kontrati. Writer SADECE bunu gorur; IdealDevice ile Memtransistor
arasindaki tek fark cihaz nesnesidir (egitim dongusu birebir ayni kalir).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class ConductanceDevice(ABC):
    """Paralel simule edilen analog bellek cihazi populasyonu (sekil == agirlik)."""

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
        """polarity in {-1,0,+1} (LTD/none/LTP), n>=0 tam sayi; G'yi yerinde gunceller."""

    @property
    @abstractmethod
    def bounds(self): ...

    @property
    @abstractmethod
    def nominal_step(self) -> float:
        """Writer'in pulse->G olceklendirmesi icin nominal ortalama LTP adimi."""

    @property
    def G(self):
        return self._G


class IdealDevice(ConductanceDevice):
    """Lineer, sinirsiz, gurultusuz referans. Baseline / 'tavan'."""

    def reset(self):
        self._G = torch.zeros(self.shape, device=self.torch_device, dtype=self.dtype)

    def read(self):
        return self._G

    def pulse(self, polarity, n):
        # her pulse tam olarak nominal_step kadar; non-idealite yok
        self._G = self._G + polarity * n.to(self.dtype) * self.nominal_step

    @property
    def bounds(self):
        return (-float("inf"), float("inf"))

    @property
    def nominal_step(self):
        return 1.0