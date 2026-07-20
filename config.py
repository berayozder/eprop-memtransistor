"""
config.py
=========
CENTRAL CONFIGURATION for the whole project.

Every tunable number and switch used anywhere in the codebase lives here, grouped
into small dataclasses. Nothing in this file "does" anything at run time -- it only
*holds* values. Every other module receives one of these config objects and reads
the fields it needs. Experiments (see run_matrix.py) are produced by taking a base
ExperimentConfig and overriding individual fields (e.g. device.kind = "ideal").

Reading tip: start here to learn *what can be varied*; then read device_interface.py
-> memtransistor.py -> synapse.py -> neurons.py -> network.py -> task.py -> train.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math


@dataclass
class NeuronConfig:
    """Parameters of the spiking neurons (LIF / ALIF) and the leaky readout.

    The network is an LSNN (Long Short-term memory Spiking Neural Network): a mix of
    plain LIF neurons and adaptive ALIF neurons. ALIF neurons have a threshold that
    rises after each spike, giving the network a slow memory needed for tasks with
    long time gaps (like evidence accumulation).
    """
    # CHECKED AGAINST BELLEC'S OFFICIAL REFERENCE CODE (github.com/IGITUGraz/
    # eligibility_propagation, Figure_3_and_S7_e_prop_tutorials/
    # tutorial_evidence_accumulation_with_alif.py): v_th, tau_m, tau_out and
    # gamma_pd below are EXACT matches to the reference's thr=0.6, tau_v=20,
    # tau_out=20, dampening_factor=0.3. tau_a and beta are NOT -- see their
    # comments below for the confirmed, quantified difference (custom design).
    n_rec: int = 120            # number of recurrent neurons in the network
    adaptive_frac: float = 0.4  # fraction of neurons that are ALIF (adaptive); 0 -> pure LIF
    v_th: float = 0.6           # firing threshold: a neuron spikes when its membrane crosses this
    tau_m: float = 20.0         # membrane time constant (ms): how fast the membrane voltage leaks
    tau_a: float = 200.0        # adaptation time constant (ms): how slowly the ALIF threshold relaxes.
                                # CUSTOM, CONFIRMED DIFFERENT: the reference uses tau_a=2000ms (10x
                                # longer) for this task -- this project's value is our own choice.
    tau_out: float = 20.0       # readout leak time constant (ms): smoothing of the output units
    beta: float = 1.6           # adaptation strength for ALIF neurons; 0 makes an ALIF behave like LIF.
                                # CUSTOM, CONFIRMED DIFFERENT: the reference does not use a flat
                                # constant here -- it DERIVES beta from tau_a/tau_m via
                                # beta_a = 1.7*(1-exp(-dt/tau_a))/(1-exp(-dt/tau_m)), which the
                                # reference's own comment calls "a heuristic value". At the
                                # reference's tau_a=2000/tau_m=20 this evaluates to ~0.0174 -- nearly
                                # 100x smaller than this project's flat 1.6. Both the VALUE and the
                                # DESIGN (flat constant vs. tau-coupled formula) differ from the
                                # reference; this project's choice is its own, not derived from theirs.
    gamma_pd: float = 0.3       # amplitude of the pseudo-derivative (surrogate gradient) used by e-prop
    dt: float = 1.0             # simulation time step (ms)

    # --- Derived decay factors (computed from the time constants above) ---
    # A leaky variable x that decays with time constant tau updates as x <- exp(-dt/tau) * x + input.
    # These properties turn the human-friendly time constants into the per-step decay multipliers.
    @property
    def alpha(self) -> float:
        """Membrane decay per step: v <- alpha * v + input."""
        return math.exp(-self.dt / self.tau_m)

    @property
    def rho(self) -> float:
        """Adaptation decay per step (ALIF): a <- rho * a + spike."""
        return math.exp(-self.dt / self.tau_a)

    @property
    def kappa(self) -> float:
        """Readout decay per step: y <- kappa * y + weighted spikes."""
        return math.exp(-self.dt / self.tau_out)


@dataclass
class TaskConfig:
    """Defines the headline problem the network must solve: Evidence Accumulation (Bellec 2020 Fig. 3).

    Evidence accumulation forces the recurrent weights -- which live on the physical
    memtransistor array -- to actively learn working memory attractors, so the
    device's physical flaws show up cleanly in accuracy.
    """
    kind: str = "evidence"          # task identifier
    n_out: int = 2                  # number of output units (left / right decision)
    seed: int = 0                   # RNG seed for generating cue sequences and noise

    # --- evidence accumulation task (Bellec 2020, Fig. 3) ---
    # CHECKED AGAINST THE OFFICIAL REFERENCE (github.com/IGITUGraz/eligibility_
    # propagation, generate_click_task_data() in tools.py): n_group/n_cues below
    # match the reference's channel-group architecture and n_cues=7 exactly.
    # cue_dur/gap/delay/decision are CUSTOM and CONFIRMED ~10x shorter overall
    # than the reference (t_cue=100, implied gap=50, delay=~1050, recall/decision
    # window=150, total ~2250 steps) -- see task.py's module docstring for the
    # full comparison. This project's compressed timing is a deliberate choice
    # (faster CPU training), not a claimed reproduction of the paper's timescale.
    n_group: int = 10               # channels per input group; groups are [left | right | recall | noise]
    n_cues: int = 7                 # number of cues per trial; ODD so there is never a left/right tie
    cue_dur: int = 15               # duration of one cue in steps
    gap: int = 5                    # empty gap between consecutive cues
    delay: int = 50                 # wait between the last cue and the decision (tests working memory)
    decision: int = 30              # length of the decision (recall) window at the end
    cue_rate: float = 0.4           # spike probability inside an active cue/recall channel.
                                    # CUSTOM, CONFIRMED DIFFERENT: the reference uses f0=0.04 (40Hz
                                    # at dt=1ms) -- 10x lower than this project's 0.4.
    noise_rate: float = 0.05        # background spike probability on all channels.
                                    # CUSTOM, CONFIRMED DIFFERENT: the reference uses f0/4=0.01 -- 5x
                                    # lower than this project's 0.05.
    p_left: float = 0.5             # probability that any given cue is a "left" cue.
                                    # CUSTOM, CONFIRMED DIFFERENT PROCESS (not just a value): the
                                    # reference does not flip a fair coin per cue. It picks a biased
                                    # per-TRIAL probability pair (0.3 vs 0.7, side randomized per
                                    # trial) and draws all n_cues cues from that -- giving each trial
                                    # a consistent lean, unlike this module's independent fair coin
                                    # per cue. See task.py's module docstring.


@dataclass
class DeviceConfig:
    """Physics parameters of the memristive device model (see memtransistor.py).

    The default numbers dp/dd/kp/kd are fitted to the real measured curve of a MoS2
    memtransistor (Sangwan 2018, Fig. 4c) by fit_device.py -- they are not invented.
    The enable_* switches let us turn individual non-idealities off one at a time
    (ablation study) to see which one is responsible for the accuracy loss.
    """
    kind: str = "memtransistor"   # which device to use: "ideal" (perfect) | "memtransistor" (realistic)
    g_min: float = 0.0            # minimum device conductance (lower bound of the state)
    g_max: float = 1.0            # window span AT THE V_G=0V FIT REFERENCE (see memtransistor.gate_gain);
                                   # the usable top of the window can extend past g_max for V_G>0V

    # Potentiation (LTP, "up") and depression (LTD, "down") step model, fitted to Fig. 4c.
    dp: float = 0.2248            # base LTP step size (fit RMSE 0.005)
    dd: float = 0.8698            # base LTD step size; dd/dp ~ 3.9 => the ABRUPT depression, the core problem
    kp: float = 1.537             # LTP nonlinearity: how quickly the up-step shrinks near the top
    kd: float = 3.762             # LTD nonlinearity: how quickly the down-step shrinks (gradual tail)

    sigma_c2c: float = 0.005      # cycle-to-cycle write noise (random jitter added to each pulse), in G units
    sigma_d2d: float = 0.05       # device-to-device variation (std of a per-device fixed step multiplier)
    read_noise: float = 0.0       # optional read noise added when the conductance is read back
    V_G: float = 50.0             # gate voltage; the Fig. 4c/4d fit reference is V_G=0V (gate_gain(0)==1.0),
                                   # so the +50V default runs at ~6.1x that reference window, not "at" it
    step_scale: float = 1.0       # multiplier on the pulse step; <1 => smaller steps => MORE states (quantization knob)

    # Ablation switches: set one to False to remove that non-ideality and isolate its effect.
    enable_nonlinearity: bool = True   # False -> constant (state-independent) step size
    enable_asymmetry: bool = True      # False -> make LTD step equal to LTP step (symmetric)
    enable_c2c: bool = True            # False -> no cycle-to-cycle write noise
    enable_d2d: bool = True            # False -> all devices identical (no device-to-device spread)


@dataclass
class SynapseConfig:
    """How network weights map onto device conductances and how updates are written."""
    w_range: float = 0.5          # a weight W in [-w_range, +w_range] maps to G in [g_min, g_max]
    writer: str = "accumulate"    # how a desired weight change becomes pulses: "direct" | "accumulate" | "verify"
    verify_max_iter: int = 5      # for writer="verify": max read-write correction iterations
    weights_on_device: tuple = ("rec", "in")  # informational: which weights live on the device (rec + in)
    differential_pair: bool = False  # True -> represent each weight as G_pos - G_neg using LTP-only pulses
    auto_refresh: bool = True        # True -> refresh a differential pair when both devices saturate together


@dataclass
class TrainConfig:
    """Training loop settings (learning rate, number of trials, feedback type, etc.)."""
    n_trials: int = 3000                # number of training trials (episodes)
    lr: float = 5e-3                    # learning rate (step size for weight updates)
    eprop_variant: str = "symmetric"    # e-prop feedback: "symmetric" (uses W_out) | "random" (fixed random)
    readout_on_device: bool = False     # False -> readout weights are ideal; True -> also stored on device
    readout_trainable: bool = True      # True -> readout learns too; False -> frozen (forces recurrent learning)
    log_every: int = 50                 # print progress every N trials
    torch_device: str = "cpu"           # compute device: "cpu" | "mps" (Apple Silicon GPU)
    seed: int = 1                       # RNG seed for weight initialization
    w_gain: float = 1.0                 # scale of the random initial weights


@dataclass
class ExperimentConfig:
    """Top-level container bundling all the sub-configs into one object.

    `field(default_factory=...)` gives every ExperimentConfig its own fresh copy of
    each sub-config (so editing one experiment's config never leaks into another).
    """
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    synapse: SynapseConfig = field(default_factory=SynapseConfig)
    train: TrainConfig = field(default_factory=TrainConfig)