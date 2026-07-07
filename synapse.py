"""
synapse.py
----------
Agirlik <-> iletkenlik kopru + PULSE writer.

W <-> G eslemesi (tek bipolar cihaz):
    W = gamma * (G - G_ref),   G_ref = pencere ortasi
    gamma = w_range / (yarim pencere)
Boylece isaretli agirlik TEK cihazdan gelir (memtransistor bipolar oldugu icin).

Writer semalari:
    direct     : istenen DeltaW'yi hemen pulse'lara cevir, yaz.
    accumulate : DeltaW'yi yuksek-hassasiyet dijital tamponda birik; sadece
                 |tampon| >= 1 pulse-degeri olunca cihaza pulse at (Demirag ruhu).
    verify     : oku-yaz kapali dongu; hedef W'ye yaklasana kadar birer pulse.
"""
from __future__ import annotations
import torch


class Synapse:
    def __init__(self, device, w_range, writer="accumulate", verify_max_iter=5):
        self.dev = device
        g_lo, g_hi = device.bounds
        # ideal cihazda sinir sonsuz -> duz esleme (W = G)
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
        self.w_step = self.gamma * device.nominal_step   # 1 pulse ~ bu kadar agirlik
        self.acc = torch.zeros(device.shape, device=device.torch_device, dtype=device.dtype)
        self.n_pulses_total = 0                          # enerji/metrik sayaci

    # -- esleme -----------------------------------------------------------
    def weight(self):
        return self.gamma * (self.dev.read() - self.g_ref)

    def init_weight(self, W):
        """Baslangic agirligini cihaz durumuna esle."""
        if self.ideal_map:
            self.dev._G = W.clone().to(self.dev.torch_device, self.dev.dtype)
        else:
            G = self.g_ref + W / self.gamma
            if hasattr(self.dev, "set_state"):
                self.dev.set_state(G)
            else:
                self.dev._G = G

    # -- yazma ------------------------------------------------------------
    def _emit(self, dW_pulses, max_pulses=30):
        """Isaretli tam sayi pulse vektorunu cihaza uygula (episode basi budce ile sinirli)."""
        dW_pulses = torch.clamp(dW_pulses, -max_pulses, max_pulses)
        polarity = torch.sign(dW_pulses)
        n = dW_pulses.abs()
        self.dev.pulse(polarity, n)
        self.n_pulses_total += int(n.sum().item())

    def update(self, desired_dW):
        # ideal cihaz: surekli, birebir agirlik guncellemesi (teorik tavan)
        if self.ideal_map:
            self.dev._G = self.dev._G + desired_dW
            return

        if self.writer == "direct":
            n = torch.round(desired_dW / self.w_step)
            self._emit(n)

        elif self.writer == "accumulate":
            self.acc = self.acc + desired_dW
            n = torch.trunc(self.acc / self.w_step)        # sadece tam pulse'lar
            self._emit(n)
            self.acc = self.acc - n * self.w_step          # kalani sakla

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