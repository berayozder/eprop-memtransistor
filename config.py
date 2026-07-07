"""
config.py
---------
Tum hiperparametreler tek yerde. Deneyler bunlardan turetilir.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math


@dataclass
class NeuronConfig:
    n_rec: int = 120            # tekrarli (recurrent) noron sayisi
    adaptive_frac: float = 0.4  # ALIF orani (0 -> saf LIF, LSNN icin 0.3-0.4)
    v_th: float = 0.6           # atesleme esigi
    tau_m: float = 20.0         # membran zaman sabiti (ms)
    tau_a: float = 200.0        # adaptasyon zaman sabiti (ms) - ALIF
    tau_out: float = 20.0       # readout leak zaman sabiti (ms)
    beta: float = 1.6           # adaptasyon gucu (ALIF). 0 -> LIF gibi.
    gamma_pd: float = 0.3       # pseudo-derivative genligi
    dt: float = 1.0             # zaman adimi (ms)

    @property
    def alpha(self) -> float:   # membran cursu
        return math.exp(-self.dt / self.tau_m)

    @property
    def rho(self) -> float:     # adaptasyon cursu
        return math.exp(-self.dt / self.tau_a)

    @property
    def kappa(self) -> float:   # readout cursu
        return math.exp(-self.dt / self.tau_out)


@dataclass
class TaskConfig:
    T: int = 500                # trial uzunlugu (adim)
    n_in: int = 20              # girdi kanali (frozen spike raster)
    input_rate: float = 0.05    # frozen girdi spike olasiligi / adim
    n_out: int = 1
    freqs: tuple = (2.0,)       # hedef = sinus toplami; debug icin tek frekans
    amps: tuple = (1.0,)
    seed: int = 0


@dataclass
class DeviceConfig:
    kind: str = "memtransistor"   # "ideal" | "memtransistor"
    g_min: float = 0.0
    g_max: float = 1.0
    # LTP/LTD taban adimlari ve nonlineerlik (Sangwan Fig. 4c'ye fit edilir)
    dp: float = 0.020             # LTP taban adimi (favorable uctaki)
    dd: float = 0.020             # LTD taban adimi
    kp: float = 2.0               # LTP nonlineerligi (saturasyon)
    kd: float = 2.0               # LTD nonlineerligi
    sigma_c2c: float = 0.005      # cycle-to-cycle yazma gurultusu (mutlak G)
    sigma_d2d: float = 0.05       # device-to-device varyasyon (adim carpani std)
    read_noise: float = 0.0       # okuma gurultusu (istege bagli)
    V_G: float = 30.0             # gate voltaji: dinamik araligi/granulariteyi ayarlar
    # ablation icin non-idealite anahtarlari
    enable_nonlinearity: bool = True
    enable_asymmetry: bool = True
    enable_c2c: bool = True
    enable_d2d: bool = True


@dataclass
class SynapseConfig:
    w_range: float = 1.0          # agirligin +-w_range araligina G [g_min,g_max]'e eslenir
    writer: str = "accumulate"    # "direct" | "accumulate" | "verify"
    verify_max_iter: int = 5      # writer="verify" icin
    weights_on_device: tuple = ("rec", "in")  # cihazda tutulan agirliklar


@dataclass
class TrainConfig:
    n_trials: int = 3000
    lr: float = 5e-3
    eprop_variant: str = "symmetric"   # "symmetric" | "random"
    readout_on_device: bool = False    # W_out ideal mi cihazda mi
    log_every: int = 50
    torch_device: str = "cpu"          # "mps" (Apple Silicon) | "cpu"
    seed: int = 1
    w_gain: float = 1.0                # baslangic agirlik olcegi


@dataclass
class ExperimentConfig:
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    synapse: SynapseConfig = field(default_factory=SynapseConfig)
    train: TrainConfig = field(default_factory=TrainConfig)