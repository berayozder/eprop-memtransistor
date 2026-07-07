"""
config.py
---------
Single source of truth for all hyperparameters.
Experiments are derived from these base configurations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math


@dataclass
class NeuronConfig:
    n_rec: int = 120            # Number of recurrent neurons
    adaptive_frac: float = 0.4  # Fraction of ALIF neurons (0 -> pure LIF, 0.3-0.4 for LSNN)
    v_th: float = 0.6           # Firing threshold
    tau_m: float = 20.0         # Membrane time constant (ms)
    tau_a: float = 200.0        # Adaptation time constant (ms) - ALIF
    tau_out: float = 20.0       # Readout leak time constant (ms)
    beta: float = 1.6           # Adaptation strength (ALIF). 0 -> acts like a LIF neuron.
    gamma_pd: float = 0.3       # Pseudo-derivative dampening factor
    dt: float = 1.0             # Simulation time step (ms)

    @property
    def alpha(self) -> float:   # Membrane leak decay factor
        return math.exp(-self.dt / self.tau_m)

    @property
    def rho(self) -> float:     # Adaptation decay factor
        return math.exp(-self.dt / self.tau_a)

    @property
    def kappa(self) -> float:   # Readout leak decay factor
        return math.exp(-self.dt / self.tau_out)


@dataclass
class TaskConfig:
    T: int = 500                # Trial length (steps)
    n_in: int = 20              # Input channels (frozen spike raster)
    input_rate: float = 0.05    # Frozen input spike probability per step
    n_out: int = 1
    freqs: tuple = (2.0,)       # Target signal frequencies (sum of sines)
    amps: tuple = (1.0,)        # Amplitudes of target signal components
    seed: int = 0


@dataclass
class DeviceConfig:
    kind: str = "memtransistor"   # "ideal" | "memtransistor"
    g_min: float = 0.0
    g_max: float = 1.0
    # LTP/LTD base step sizes and nonlinearity parameters (fitted to Sangwan Fig. 4c)
    dp: float = 0.020             # LTP base step size (at the favorable end)
    dd: float = 0.020             # LTD base step size (at the favorable end)
    kp: float = 2.0               # LTP nonlinearity exponent (saturation)
    kd: float = 2.0               # LTD nonlinearity exponent (saturation)
    sigma_c2c: float = 0.005      # Cycle-to-cycle write noise (absolute conductance G)
    sigma_d2d: float = 0.05       # Device-to-device variation (step scaling factor std)
    read_noise: float = 0.0       # Read noise standard deviation
    V_G: float = 30.0             # Gate voltage: tunes dynamic range and programming granularity
    # Ablation keys for non-idealities
    enable_nonlinearity: bool = True
    enable_asymmetry: bool = True
    enable_c2c: bool = True
    enable_d2d: bool = True


@dataclass
class SynapseConfig:
    w_range: float = 1.0          # Synaptic weight scale, mapped to conductance G [g_min, g_max]
    writer: str = "accumulate"    # Programming scheme: "direct" | "accumulate" | "verify"
    verify_max_iter: int = 5      # Maximum loop iterations for write-verify scheme
    weights_on_device: tuple = ("rec", "in")  # Synaptic weights stored on memristive hardware


@dataclass
class TrainConfig:
    n_trials: int = 3000
    lr: float = 5e-3
    eprop_variant: str = "symmetric"   # "symmetric" | "random" (weight-transport-free)
    readout_on_device: bool = False    # W_out simulated ideally or placed on device
    log_every: int = 50
    torch_device: str = "cpu"          # "mps" (Apple Silicon) | "cuda" | "cpu"
    seed: int = 1
    w_gain: float = 1.0                # Initial weight gain/scale factor


@dataclass
class ExperimentConfig:
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    synapse: SynapseConfig = field(default_factory=SynapseConfig)
    train: TrainConfig = field(default_factory=TrainConfig)