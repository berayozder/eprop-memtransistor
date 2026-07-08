# e-prop on MoS₂ Memtransistor (Device-in-the-Loop Simulation)

A simulation embedding the synaptic weights of the e-prop learning rule (Bellec et al., 2020) onto a phenomenological MoS₂ memtransistor model (Sangwan et al., 2018) via **pulse-based + gradient-accumulation** programming.

**Headline Task: Evidence Accumulation** (Bellec 2020, Fig. 3) — The network must accumulate randomized left/right cues over time and select the majority side in the decision window. Recurrent plasticity is REQUIRED (each trial is generated dynamically, demanding working memory and counting). This ensures the device is actively in the loop.
**Debug Task: Pattern Generation** — A reservoir-like setup predicting a sum of sines, used for quick sanity checks.

---

## Data Flow

```
LSNN forward pass (network.py)
   └─ online e-prop traces → desired update ΔW = -lr · Σ_t L · ē
        └─ Synapse.update (synapse.py)   [writer: direct / accumulate / verify]
             └─ polarity + n pulses
                  └─ Memtransistor.pulse (memtransistor.py)  → G updated (non-ideal)
       W_rec, W_in = γ · (G - G_ref)   ← read by the next trial forward pass
   Readout (W_out, b_out): updated ideally (or on-device optionally)
```

`IdealDevice` acts as a continuous, noiseless, and unbounded reference (the theoretical ceiling). The training loop remains identical for both device variants, enabling a clean and fair ablation.

---

## Repository Files

- [config.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/config.py) — Single source of truth for all configurations (dataclasses).
- [device_interface.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/device_interface.py) — `ConductanceDevice` abstraction and `IdealDevice` implementation.
- [memtransistor.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/memtransistor.py) — MoS₂ model: bipolar, state-dependent saturating LTP/LTD (Fig. 4c), V_G resolution tuning, and ablatable non-idealities.
- [synapse.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/synapse.py) — Agirlik <-> iletkenlik mapping bridge and pulse writer options (accumulate, direct, verify).
- [neurons.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/neurons.py) — Neural primitives and pseudo-derivative calculation.
- [network.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/network.py) — LSNN (LIF+ALIF) forward dynamics and online e-prop gradient accumulation.
- [task.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/task.py) — Task generation helpers for evidence accumulation and pattern generation.
- [train.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/train.py) — SNN training loop and trial logging.
- [run_experiment.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/run_experiment.py) — Baseline comparison (ideal vs. memtransistor) for evidence curves.
- [fit_device.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/fit_device.py) — Scipy least-squares fitting of LTP/LTD parameters to Sangwan 2018 (Fig. 4c).
- [gradcheck.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/gradcheck.py) — Mathematical correctness verification of e-prop gradients against autograd/BPTT.
- [run_matrix.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/run_matrix.py) — Multi-seed matrix harness for V_G sweeps, ablation, and quantization studies.

---

## Getting Started

Install the required packages and run the scripts:

```bash
pip install torch matplotlib scipy pandas openpyxl
python fit_device.py            # Fits Fig. 4c data -> generates device_fit.png
python gradcheck.py             # Validates implementation correctness (all tests PASS)
python run_experiment.py        # Compares ideal vs. memtransistor -> generates evidence_curves.png
python run_matrix.py            # Runs multi-seed sweeps (light mode) -> generates matrix_figures.png
python run_matrix.py --full     # Full scale paper sweeps (slower, recommended with GPU/MPS)
```

For Apple Silicon Mac, you can set `TrainConfig.torch_device = "mps"` in `config.py`.
Note: The `light=True` setting in `run_experiment.py` is for fast verification. For the full paper parameters, set `light=False` (n_rec=120, 7 cues, 3000 trials).

---

## Configurable Sweep Options (config.py)

- **V_G Sweep:** Adjusts `DeviceConfig.V_G` ∈ {-50, -25, 0, 25, 50} to evaluate how resolution/switching ratio trends affect SNN learning accuracy.
- **Writer comparison:** Toggle `SynapseConfig.writer` = "direct" | "accumulate" | "verify".
- **Non-ideality ablation:** Set `DeviceConfig.enable_{nonlinearity, asymmetry, c2c, d2d}` to False individually to pinpoint the dominant physical bottleneck.
- **eprop variant:** Choose `TrainConfig.eprop_variant` = "symmetric" | "random" (for hardware-friendly random feedback, avoiding weight-transport).
- **On-device Readout:** Enable `TrainConfig.readout_on_device = True` to allow device noise to propagate back through symmetric feedback.
- **ALIF vs. LIF:** Adjust `NeuronConfig.adaptive_frac` = 0.4 (mixed LSNN) ↔ 0.0 (pure LIF). Adaptivity is crucial for long-term working memory and counting.

---

## Critical Findings and Verification Results

- **Device Fit (Locked to Fig. 4c):** Fitted parameters are dp=0.2248, kp=1.537 (LTP, RMSE 0.005) and dd=0.8698, kd=3.762 (LTD, RMSE 0.012). Conductance saturates rapidly within ~10 pulses.
- **Strong LTP/LTD Asymmetry (Physical):** Potentiation is gradual (~10 steps), but depression is sudden (large single-pulse drop, dd/dp ~ 3.9). This asymmetry is the most disruptive non-ideality for e-prop.
- **Ablation Insight:** Disabling asymmetry brings accuracy back up significantly, confirming that asymmetric switching is the dominant performance bottleneck. This motivates mitigations like differential pairs or gated tuning.
- **Gradient Verification (gradcheck.py):** All checks pass. Feedforward e-prop equals BPTT (relative error ~2e-7), LIF recurrent factorization matches BPTT exactly (relative error ~7e-8), and recurrent cosine similarity shows positive gradient alignment (~0.53) indicating a valid descent direction.

---

## Modeling Assumptions to Declare in the Manuscript

1. **Episodic e-prop:** Gradients are accumulated over time and programmed at the end of each trial. This represents an episodic weight writer constraint rather than a fully online real-time update.
2. **Phenomenological Device:** Fits pulse-response directly to Fig. 4c. Step decay near the boundaries captures the essence of the physical window function.
3. **V_G Mapping:** switching ratio (SR) translates directly to the effective number of conductance states (pencere span/resolution), calibrated to the ~37x experimental span.
4. **Single-Device Synapse:** A single bipolar memtransistor encodes signed weights (avoiding differential pairs).