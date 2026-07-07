"""
memtransistor.py
----------------
MoS2 memtransistor'un FENOMENOLOJIK modeli (Sangwan 2018).
  - Bipolar: pozitif pulse LTP, negatif pulse LTD (Fig. 4c).
  - Pulse-based: her pulse mevcut duruma bagli, nonlineer, asimetrik, saturasyonlu
    bir DeltaG uygular. n pulse = n ardisik durum guncellemesi (fizik boyle).
  - V_G: dinamik araligi ve guncelleme granularitesini ayarlar (Fig. 2b inset:
    switching orani V_G ile ~300'den ~8'e degisiyor). Fenomenolojik olarak
    kullanilabilir pencereyi ve adim boyunu olcekler.
  - Non-idealiteler ablation icin ayri ayri kapatilabilir.

Durum-bagli adim (saturasyonlu ustel, Fig. 4c bicimi):
  LTP:  G <- G + dp * exp(-kp * (G - g_lo)/(g_hi - g_lo))     (g_lo yakininda buyuk, g_hi yakininda kucuk)
  LTD:  G <- G - dd * exp(-kd * (g_hi - G)/(g_hi - g_lo))
Sinir yakininda adim dogal olarak sonuyor -> Sangwan pencere fonksiyonunun (denk.17) ozu.
"""
from __future__ import annotations
import math
import torch
from device_interface import ConductanceDevice


def gate_gain(V_G: float) -> float:
    """V_G -> kullanilabilir dinamik aralik / granularite carpani (0,1].
    Fenomenolojik: yuksek V_G -> daha genis pencere, daha cok ayirt edilebilir durum.
    V_G ~ [-50, 50] -> ~[0.15, 1.0] (Fig. 2b switching orani egilimini yansitir)."""
    return 0.15 + 0.85 * (1.0 / (1.0 + math.exp(-0.08 * V_G)))


class Memtransistor(ConductanceDevice):
    def __init__(self, shape, cfg, torch_device="cpu", dtype=torch.float32, seed=0):
        self.cfg = cfg
        super().__init__(shape, torch_device=torch_device, dtype=dtype, seed=seed)

    # -- kurulum ----------------------------------------------------------
    def reset(self):
        c = self.cfg
        g = gate_gain(c.V_G)
        # V_G kullanilabilir pencereyi daraltir (dusuk V_G -> kucuk switching orani)
        self.g_lo = c.g_min
        self.g_hi = c.g_min + (c.g_max - c.g_min) * g
        self._gate = g

        # device-to-device: her cihaza sabit bir adim-carpani (reset'te cekilir)
        if c.enable_d2d and c.sigma_d2d > 0:
            noise = torch.randn(self.shape, generator=self.gen, dtype=self.dtype)
            self._d2d = (1.0 + c.sigma_d2d * noise).clamp(min=0.3).to(self.torch_device)
        else:
            self._d2d = torch.ones(self.shape, device=self.torch_device, dtype=self.dtype)

        # orta noktadan basla (isaretli agirlik icin G_ref); Synapse init'te uzerine yazabilir
        mid = 0.5 * (self.g_lo + self.g_hi)
        self._G = torch.full(self.shape, mid, device=self.torch_device, dtype=self.dtype)

    def set_state(self, G):
        """Synapse baslangic agirligini G'ye eslerken kullanir."""
        self._G = G.clamp(self.g_lo, self.g_hi).to(self.torch_device, self.dtype)

    # -- okuma ------------------------------------------------------------
    def read(self):
        if self.cfg.read_noise > 0:
            n = torch.randn(self.shape, generator=self.gen, dtype=self.dtype).to(self.torch_device)
            return (self._G + self.cfg.read_noise * n).clamp(self.g_lo, self.g_hi)
        return self._G

    # -- tek pulse'luk durum-bagli DeltaG ---------------------------------
    def _delta(self, polarity):
        """polarity: +1 LTP, -1 LTD, 0 yok. Sekil == shape. Tek pulse'luk DeltaG."""
        c = self.cfg
        span = max(self.g_hi - self.g_lo, 1e-8)
        gnorm = (self._G - self.g_lo) / span          # [0,1]

        if c.enable_nonlinearity:
            ltp = c.dp * torch.exp(-c.kp * gnorm)         # g_lo'da buyuk, g_hi'de kucuk
            ltd = c.dd * torch.exp(-c.kd * (1.0 - gnorm)) # g_hi'de buyuk, g_lo'da kucuk
        else:
            ltp = torch.full_like(self._G, c.dp)
            ltd = torch.full_like(self._G, c.dd)

        if not c.enable_asymmetry:
            # asimetriyi kapat: LTD adimini LTP ile ayni yap
            ltd = ltp.clone()

        # gate granularitesi: dusuk V_G -> kucuk pencere -> kucuk mutlak adim
        ltp = ltp * self._gate * self._d2d
        ltd = ltd * self._gate * self._d2d

        step = torch.where(polarity > 0, ltp, torch.where(polarity < 0, -ltd, torch.zeros_like(ltp)))
        return step

    def pulse(self, polarity, n):
        """n pulse = n ardisik durum-bagli guncelleme (vektorize, max(n) kez dongu)."""
        c = self.cfg
        n = n.to(torch.int64)
        max_n = int(n.max().item()) if n.numel() else 0
        for k in range(max_n):
            active = (n > k)                      # bu turda hala pulse alacak cihazlar
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
        # writer olceklendirmesi: pencere ortasindaki ortalama LTP adimi
        return 0.5 * (self.cfg.dp + self.cfg.dd) * self._gate * math.exp(-0.5 * self.cfg.kp)