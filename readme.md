# e-prop on MoS₂ Memtransistor (device-in-the-loop simulation)

A simulator that maps the weights of the e-prop learning rule (Bellec et al., 2020) onto a phenomenological MoS₂ memtransistor model (Sangwan et al., 2018) using **pulse-based + gradient-accumulation** programming. Task: spatiotemporal pattern generation.

## Data Flow

```
LSNN forward (network.py)
   └─ online e-prop traces → desired ΔW = -lr · Σ_t L · ebar
        └─ Synapse.update (synapse.py)   [writer scheme: direct / accumulate / verify]
             └─ polarity + n pulses
                  └─ Memtransistor.pulse (memtransistor.py)  → conductance G is updated (non-ideal)
       W_rec, W_in = γ · (G - G_ref)   ← read in the next trial forward pass
   readout (W_out, b_out): ideal (or on-device), updated directly
```

`IdealDevice` = continuous, noise-free, unbounded reference (theoretical ceiling). The execution loop is identical for both configurations; the only difference is the device object instance, allowing fair ablation studies.

## File Structure
- `config.py` — Hyperparameters represented as configurations (dataclass).
- `device_interface.py` — `ConductanceDevice` abstract base class + `IdealDevice` reference implementation.
- `memtransistor.py` — Phenomenological MoS₂ device model: bipolar, state-dependent saturating LTP/LTD (fitting Fig. 4c), V_G gate voltage control, and ablatable non-idealities.
- `synapse.py` — W↔G mapping bridge + pulse writer schemes (defaults to accumulate).
- `neurons.py` — Heaviside activation and pseudo-derivative calculation helper.
- `network.py` — LSNN (LIF+ALIF) class + online e-prop implementation (traces, learning signals, and gradients).
- `task.py` — Pattern generation task data generation (frozen Poisson inputs → target wave).
- `train.py` — Episodic e-prop training loop.
- `run_experiment.py` — Compares the ideal baseline vs. memtransistor-in-the-loop, generating output curves.

## How to Run
```bash
pip install torch matplotlib scipy
python run_experiment.py        # Generates learning_curves.png
```
Apple Silicon: Set `TrainConfig.torch_device = "mps"` in `config.py`.

## Planned Experiments (Configuration Sweeps)
- **V_G Sweep (Headline):** Sweep `DeviceConfig.V_G` ∈ {-30, 0, 20, 40} → tunes dynamic range / granularity vs. SNN learning accuracy.
- **Writer Comparison:** Compare `SynapseConfig.writer` = "direct" | "accumulate" | "verify".
- **Non-ideality Ablation:** Toggle `DeviceConfig.enable_{nonlinearity, asymmetry, c2c, d2d}` individually to build a degradation cost table.
- **e-prop Variants:** Compare `TrainConfig.eprop_variant` = "symmetric" | "random" (weight-transport-free, hardware-plausible).
- **On-Device Readout:** Set `TrainConfig.readout_on_device = True` (future step).
- **Neuron Heterogeneity:** Vary `NeuronConfig.adaptive_frac` = 0 (pure LIF) ↔ 0.4 (LSNN).

## Modeling Assumptions to Highlight in the Manuscript
1. **Episodic e-prop:** Gradients are accumulated over time step 't' during a trial and written at the end of the trial (instead of true per-step online updates). This is a design choice that maps directly onto gradient-accumulation hardware writers.
2. **Phenomenological device model:** The LTP/LTD pulse-response is fit directly to Sangwan Fig. 4c. Step-size decay near boundaries captures the core function of the Sangwan window function.
3. **V_G mapping** (`gate_gain`) is phenomenological, representing a monotonic curve modeling the switching-ratio trend from Fig. 2b.
4. Single-device bipolar signed weight mapping (assumes single device per synapse instead of differential pairs).

## Next Steps
- Quantitatively fit `memtransistor.py` parameters (dp, dd, kp, kd, V_G) to experimental MoS2 Fig. 4c LTP/LTD data.
- Transition readout layer update logic onto on-device synapses.
- Validate e-prop gradient calculations against autograd/BPTT baseline in a small LIF test.