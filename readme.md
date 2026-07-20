# e-prop on MoS₂ Memtransistor (Device-in-the-Loop Simulation)

A simulation embedding the synaptic weights of the e-prop learning rule (Bellec et al., 2020) onto a phenomenological MoS₂ memtransistor model (Sangwan et al., 2018) via **pulse-based + gradient-accumulation** programming.

**Headline Task: Evidence Accumulation** (Bellec 2020, Fig. 3) — The network must accumulate randomized left/right cues over time and select the majority side in the decision window. Recurrent plasticity is REQUIRED (each trial is generated dynamically, demanding working memory and counting). This ensures the device is actively in the loop.

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
- [synapse.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/synapse.py) — Weight <-> conductance mapping bridge and pulse writer options (accumulate, direct, verify).
- [neurons.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/neurons.py) — Neural primitives and pseudo-derivative calculation.
- [network.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/network.py) — LSNN (LIF+ALIF) forward dynamics and online e-prop gradient accumulation.
- [task.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/task.py) — Dynamic task generator for the Evidence Accumulation problem (Bellec 2020 Fig. 3).
- [train.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/train.py) — SNN training loop and trial logging.
- [run_experiment.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/run_experiment.py) — Baseline comparison (ideal vs. memtransistor) for evidence curves.
- [fit_device.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/fit_device.py) — Scipy least-squares fitting of LTP/LTD parameters to Sangwan 2018 (Fig. 4c).
- [gradcheck.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/gradcheck.py) — Mathematical correctness verification of e-prop gradients against autograd/BPTT.
- [run_matrix.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/run_matrix.py) — Multi-seed matrix harness for V_G sweeps, ablation, and quantization studies.
- [physics_vg_check.py](file:///Users/berayozder/Desktop/Side_projects/memtransistor_eprop/physics_vg_check.py) — Standalone cross-check: derives V_G's effect on the LTP/LTD curve directly from Sangwan 2018's transport-physics equations and compares it to `gate_gain()`'s heuristic (see "Physics cross-check" below).

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
python physics_vg_check.py      # Standalone physics-vs-heuristic V_G cross-check (see below)
```

For Apple Silicon Mac, you can set `TrainConfig.torch_device = "mps"` in `config.py`.
Note: The `light=True` setting in `run_experiment.py` is for fast verification. Setting `light=False` (n_rec=120, 7 cues, 3000 trials) matches the paper's network size and cue count, but **not** its timescale — checked directly against the official reference code ([IGITUGraz/eligibility_propagation](https://github.com/IGITUGraz/eligibility_propagation)): this project's cue/gap/delay/decision durations and input spike rates are a deliberate ~10x-shorter compression of the reference's, for faster CPU-only training (see `task.py` and `config.py` for the exact, quantified comparison). `v_th`, `tau_m`, `tau_out`, `gamma_pd`, and `n_cues` do match the reference exactly; `tau_a`, `beta`, and the cue-generation process (fair coin vs. the reference's per-trial biased draw) do not.

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

## Physics Cross-Check: Does V_G Rescale the Window, or the Step Size?

`memtransistor.py`'s `gate_gain(V_G)` heuristic keeps the LTP/LTD step size (`dp`/`dd`/`kp`/`kd`, fitted once to Fig. 4c/4d) fixed and only rescales the usable conductance *window* with V_G — i.e. it assumes V_G only ever adds more distinguishable states. `physics_vg_check.py` tests this assumption directly: it implements the paper's own transport-physics equations (Methods Eqs. 1, 2, 16, 17 — a single unified formula for both pulse polarities, not the full split V_D≥0/V_D<0 branches of Eqs. 18-22), fits the free device-specific constants once at V_G=0V (Fig. 4c/4d's actual measurement condition), and then holds those fixed while sweeping V_G to see what the equations themselves predict.

**Finding (qualitative, not a precise quantitative prediction — see caveats below):** the paper's drain-current equation has an explicit gate-voltage term (`exp[e(V_G-V_th)/(c_t k_B T)]`) that multiplies I_D directly, and the state update is `dw/dt = μ·R_on·I_D(t)·F(w)` — so, taken at face value, V_G rescales the *rate* w moves per pulse (a step-size effect), not the window shape the heuristic assumes. `physics_vg_check.py` reports this via a bound-free "analytic step-size proxy" (|dw/dt|·pulse-duration at a fixed reference state) rather than by re-fitting `dp`/`dd` directly, because the naive re-fit approach turned out to have real problems (see below). Qualitatively: the model predicts the LTP branch is essentially off at V_G=-50V (consistent with the paper's own note that "the forward-biased device is completely off at V_G=-50V") and saturates within a single pulse by V_G≥+25V — a threshold-like, much larger swing than the heuristic's smooth ~6x window-only change.

**Why the specific numbers should NOT be over-trusted (found on review of the first version of this check):**
- **The headline ratio is algebra, not a fit result.** The step-size ratio between any two V_G values reduces exactly to `exp(ΔV_G/(c_t·k_BT))` — every parameter actually fit to Fig. 4c/4d's data (current scale, rate constant, barrier sensitivity, initial state) cancels out of it. So the "~10²⁰x over 100V" figure the script prints is a restatement of the paper's own `c_t=83.3` constant — **measured on a different device** (Fig. 2a, not Fig. 4c/4d) — extrapolated over the sweep range, not something learned from Fig. 4c/4d at all. A ratio that size also isn't physically plausible for a real device (some unmodeled floor — leakage, access resistance — would dominate long before 20 orders of magnitude); it should be read as "the paper's own subthreshold-FET physics, with no floor, implies an effect far too large to be just a window effect," not as a literal prediction.
- **Re-fitting `dp`/`dd` to the predicted curves is illustrative only, and several points are bound-censored** (the LTP branch is predicted "off" or "saturates in one pulse" at the sweep's extremes, so the re-fit optimizer lands exactly on its own bounds — a censored estimate, not a converged one). `vg_heuristic_vs_physics.png` marks these hollow.
- **The single-formula design cannot represent genuine LTP≠LTD asymmetry** (one set of constants, direction set only by the pulse's sign) — so any *difference* between re-fit `dp` and `dd` reflects where the branch-split falls on a symmetric trajectory, not a physics-predicted asymmetry.
- **Fit quality is honestly worse than the phenomenological model on a like-for-like basis**: RMSE 0.148 μA / 0.349 normalized, vs. `fit_device.py`'s 0.005 (LTP) / 0.012 (LTD) normalized — expected, since this MVP uses one branch instead of two and drops the tunneling/second image-charge terms.
- **A multi-start robustness check (6 starts) converges consistently** on the V_G=0V fit itself (RMSE within [0.148, 0.196] μA), but per the first point above, that consistency does not extend to validating the headline ratio, which never depended on the fit to begin with.

**Implication:** treat this as evidence that the current `gate_gain()` heuristic's window-only, step-flat design has no equation-level support from the paper — not as a validated replacement law. The honest takeaway is qualitative (V_G plausibly behaves more like a threshold/rate knob than a pure resolution knob), and the biggest open uncertainty is whether `c_t` (and the rest of the borrowed Fig. 2a constants) transfer to the Fig. 4c/4d device at all — unverifiable from the data in this paper. See `physics_vg_check.py`'s module docstring ("HONEST LIMITATIONS", six numbered points) for the full derivation and caveats. This remains a standalone device-curve-level check only — not wired into the simulator or `run_matrix.py`'s task-accuracy experiments.

---

## Modeling Assumptions to Declare in the Manuscript

1. **Episodic e-prop:** Gradients are accumulated over time and programmed at the end of each trial. This represents an episodic weight writer constraint rather than a fully online real-time update.
2. **Phenomenological Device:** Fits pulse-response directly to Fig. 4c. Step decay near the boundaries captures the essence of the physical window function.
3. **V_G Mapping:** switching ratio (SR) translates directly to the effective number of conductance states (window span/resolution), normalized relative to V_G=0V — the actual measurement condition of Fig. 4c/4d ("V_G = 0 V for all measurements in c, d"), not +50V as an earlier version of this doc assumed. `gate_gain(0.0) == 1.0` exactly; V_G=+50V gives ~6.1x that reference window, V_G=-50V gives ~0.16x. See also the "Physics cross-check" section below, which finds this window-only picture likely understates V_G's effect on the underlying step size.
4. **Single-Device Synapse:** A single bipolar memtransistor encodes signed weights (avoiding differential pairs); the two-device differential-pair mitigation is also implemented and compared (`synapse.differential_pair`).
5. **No refractory period / unit transmission delay:** neurons can spike on consecutive steps (the paper's Methods mentions a 2-5 ms refractory period in some of their simulations, which is not modeled here); the 1 ms simulation step is also taken as the transmission delay `d` in the eligibility-trace equations, matching the paper's default (not the "more realistic delay `d`" variant it mentions as an extension).
6. **Scope vs. the paper:** only the supervised, evidence-accumulation use of e-prop (symmetric and random feedback) is implemented. The paper's "adaptive e-prop" (where the broadcast weights `B` themselves evolve) and reward-based e-prop (deep RL / Atari) are not implemented here — out of scope for this device-in-the-loop study.