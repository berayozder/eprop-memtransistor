"""
network.py
==========
THE HEART OF THE SYSTEM: the spiking network's forward pass AND the e-prop learning
rule, both in one place.

The network is an LSNN (mix of LIF and adaptive ALIF neurons). For each trial it:
  1) runs the neurons forward in time, reading its weights from the device,
  2) computes, online, how much each weight should change (the e-prop gradient).

WHY E-PROP? Standard backprop-through-time (BPTT) would require storing every past
state and walking backward -- not biologically plausible and not hardware-friendly.
e-prop (Bellec 2020) factorizes the gradient into a *local eligibility trace* (kept
per synapse, forward in time) times an *instantaneous learning signal*. It drops one
term of the true gradient (the neuron-to-neuron backward path), so it is an
approximation -- but a good one, and it is exactly what gradcheck.py verifies.

Map of the e-prop quantities computed below (Bellec 2020 equation numbers):
  - membrane / spike:      v, z              (eqs. 6,7,9,10)
  - pseudo-derivative:     psi               (Methods; see neurons.py)
  - eligibility vector:    zbar (= eps_v = low-pass of presynaptic spikes),
                           eps_a (extra ALIF adaptation component, eq. 24)
  - eligibility trace:     e = psi * (zbar - beta * eps_a)      (eq. 25)
  - filtered trace:        ebar = low-pass_kappa(e)             (eq. 28, readout leak)
  - learning signal:       L = (y - y*) fed back through B      (eq. 4)
  - gradient:              grad_W += L * ebar   (summed over time t)

The gradient is accumulated over the whole trial and applied once at the end
("episodic" e-prop) -- a modelling choice that matches the accumulate writer.
"""
from __future__ import annotations
import torch
from neurons import heaviside, pseudo_derivative


class LSNN:
    """A recurrent spiking network (LIF+ALIF) whose weights live on devices, trained by e-prop."""

    def __init__(self, ncfg, tcfg, syn_rec, syn_in, readout, b_out,
                 variant="symmetric", B_fixed=None,
                 torch_device="cpu", dtype=torch.float32):
        """
        Args:
            ncfg, tcfg: neuron and task configs.
            syn_rec: Synapse holding the recurrent weights W_rec (on device).
            syn_in:  Synapse holding the input weights W_in (on device).
            readout: either a plain tensor [n_out, n_rec] (ideal readout) or a Synapse.
            b_out:   readout bias [n_out] (always ideal).
            variant: "symmetric" (feedback = W_out) or "random" (feedback = fixed B_fixed).
            B_fixed: fixed random feedback matrix for the "random" variant.
        """
        self.nc, self.tc = ncfg, tcfg
        self.syn_rec = syn_rec      # recurrent-weight synapse (on device)
        self.syn_in = syn_in        # input-weight synapse (on device)
        self.readout = readout      # readout weights: plain tensor (ideal) or Synapse (on device)
        self.readout_is_device = hasattr(readout, "weight")   # True if readout is a Synapse
        self.b_out = b_out          # readout bias [n_out] (always ideal)
        self.variant = variant      # e-prop feedback type
        self.B_fixed = B_fixed      # fixed feedback for the "random" variant
        self.dev = torch_device
        self.dtype = dtype

        n = ncfg.n_rec
        # Decide which neurons are adaptive (ALIF): the first `adaptive_frac` fraction.
        n_ad = int(round(ncfg.adaptive_frac * n))
        self.is_adaptive = torch.zeros(n, device=torch_device, dtype=dtype)
        self.is_adaptive[:n_ad] = 1.0
        # Per-neuron adaptation strength: beta for ALIF neurons, 0 for LIF neurons.
        self.beta_vec = ncfg.beta * self.is_adaptive

    def run_trial(self, X, Ystar, mask=None, loss="regression", accumulate_grads=True,
                  return_traces=False):
        """Run one trial: forward pass + online e-prop gradient accumulation.

        Args:
            X:      [T, n_in] input spike sequence (frozen for this trial).
            Ystar:  [T, n_out] target output.
            mask:   [T] optional 0/1 mask marking which steps count for loss/gradient
                    (the decision window for evidence accumulation).
            loss:   "classification" (softmax cross-entropy) or "regression" (MSE).
            accumulate_grads: whether to compute e-prop gradients (True during training).
            return_traces:    if True, also return the raw eligibility traces (for gradcheck).
        Returns:
            dict with loss, outputs y, and gradients grad_rec/grad_in/grad_out/grad_b.
        """
        nc, tc = self.nc, self.tc
        alpha, rho, kappa = nc.alpha, nc.rho, nc.kappa   # membrane / adaptation / readout decays
        v_th, beta = nc.v_th, self.beta_vec
        n, n_in, n_out = nc.n_rec, tc.n_in, tc.n_out
        T = X.shape[0]
        dev, dt = self.dev, self.dtype

        # --- Read the weights from the device(s) for this trial ---
        W_rec = self.syn_rec.weight()                       # [n, n] recurrent weights
        W_rec = W_rec - torch.diag(torch.diagonal(W_rec))   # remove self-connections (zero diagonal)
        W_in = self.syn_in.weight()                         # [n, n_in] input weights
        # Readout weights: read from device (noisy) if on device, else the plain tensor.
        W_out = self.readout.weight() if self.readout_is_device else self.readout
        # Feedback matrix for the learning signal: W_out (symmetric) or fixed random B.
        B = W_out if self.variant == "symmetric" else self.B_fixed

        # --- Neuron state variables (all start at zero) ---
        z = torch.zeros(n, device=dev, dtype=dt)             # spikes at previous step z^{t-1}
        v = torch.zeros(n, device=dev, dtype=dt)             # membrane voltage
        a = torch.zeros(n, device=dev, dtype=dt)             # ALIF adaptation variable
        y = torch.zeros(n_out, device=dev, dtype=dt)         # leaky readout output

        # --- e-prop trace variables (per synapse) ---
        zbar_rec = torch.zeros(n, device=dev, dtype=dt)      # eps_v for recurrent presynaptic spikes
        zbar_in = torch.zeros(n_in, device=dev, dtype=dt)    # eps_v for input presynaptic spikes
        epsa_rec = torch.zeros(n, n, device=dev, dtype=dt)   # eps_a (ALIF component) [post, pre]
        epsa_in = torch.zeros(n, n_in, device=dev, dtype=dt)
        ebar_rec = torch.zeros(n, n, device=dev, dtype=dt)   # kappa-filtered eligibility (recurrent)
        ebar_in = torch.zeros(n, n_in, device=dev, dtype=dt) # kappa-filtered eligibility (input)
        zbar_out = torch.zeros(n, device=dev, dtype=dt)      # kappa-filtered spikes for readout grad
        bbar_out = 0.0                                        # kappa-filtered constant for bias grad

        # --- Gradient accumulators (summed over time) ---
        grad_rec = torch.zeros(n, n, device=dev, dtype=dt)
        grad_in = torch.zeros(n, n_in, device=dev, dtype=dt)
        grad_out = torch.zeros(n_out, n, device=dev, dtype=dt)
        grad_b = torch.zeros(n_out, device=dev, dtype=dt)

        loss_acc = 0.0                                       # running loss sum
        n_mask = 0                                           # number of counted (masked) steps
        ys = torch.zeros(T, n_out, device=dev, dtype=dt)     # stored outputs for all steps
        e_rec_list = [] if return_traces else None           # raw eligibility traces (gradcheck only)
        e_in_list = [] if return_traces else None

        for t in range(T):
            x_t = X[t]                                        # input at this step [n_in]

            # --- Membrane update and spike (uses previous spikes z) ---
            I = W_rec @ z + W_in @ x_t                        # total input current [n]
            v = alpha * v + I - z * v_th                      # leaky membrane with reset-by-previous-spike
            A = v_th + beta * a                               # effective threshold (ALIF raises it; LIF: A=v_th)
            z_new = heaviside(v - A)                          # spike if membrane crosses threshold
            psi = pseudo_derivative(v, A, v_th, nc.gamma_pd)  # surrogate gradient of the spike [n]

            # --- Eligibility vectors (low-pass of presynaptic activity) ---
            zbar_rec = alpha * zbar_rec + z                   # low-pass of z^{t-1} (recurrent pre)
            zbar_in = alpha * zbar_in + x_t                   # low-pass of the input
            # Eligibility trace (eq. 25): psi times (pre-trace minus the ALIF adaptation part).
            e_rec = psi[:, None] * (zbar_rec[None, :] - beta[:, None] * epsa_rec)
            e_in = psi[:, None] * (zbar_in[None, :] - beta[:, None] * epsa_in)
            if return_traces:
                e_rec_list.append(e_rec.clone())
                e_in_list.append(e_in.clone())
            # Update the ALIF adaptation eligibility eps_a (eq. 24).
            epsa_rec = psi[:, None] * zbar_rec[None, :] + (rho - beta[:, None] * psi[:, None]) * epsa_rec
            epsa_in = psi[:, None] * zbar_in[None, :] + (rho - beta[:, None] * psi[:, None]) * epsa_in
            # Filter the eligibility with the readout leak kappa (eq. 28).
            ebar_rec = kappa * ebar_rec + e_rec
            ebar_in = kappa * ebar_in + e_in

            # --- Readout and error ---
            y = kappa * y + W_out @ z_new + self.b_out        # leaky readout [n_out] (logits or regression)
            ys[t] = y
            m_t = 1.0 if mask is None else float(mask[t])     # is this step counted? (decision window)

            if loss == "classification":
                # Softmax + cross-entropy, ONLY on masked steps (the decision window).
                ex = torch.exp(y - y.max())                   # numerically stable softmax
                pi = ex / ex.sum()
                err = (pi - Ystar[t]) * m_t                   # gradient of CE w.r.t. logits (masked)
                if m_t > 0:
                    loss_acc += -float((Ystar[t] * torch.log(pi + 1e-9)).sum().item())
                    n_mask += 1
            else:
                err = (y - Ystar[t])                          # regression error (MSE gradient)
                loss_acc += 0.5 * float((err ** 2).sum().item())
                n_mask += 1

            if accumulate_grads:
                # Learning signal L: project the output error back through the feedback matrix B.
                L = B.t() @ err                               # [n]
                # Recurrent / input weight gradients: learning signal times filtered eligibility.
                grad_rec += L[:, None] * ebar_rec
                grad_in += L[:, None] * ebar_in
                # Readout weight gradient: error times the kappa-filtered presynaptic spikes.
                zbar_out = kappa * zbar_out + z_new
                grad_out += err[:, None] * zbar_out[None, :]
                # Bias enters the readout inside the kappa recursion (y = kappa*y + ... + b),
                # so its gradient must be kappa-filtered too: dy_t/db = low-pass_kappa(1).
                bbar_out = kappa * bbar_out + 1.0
                grad_b += err * bbar_out

            a = rho * a + z_new                               # update ALIF adaptation (eq. 10)
            z = z_new                                         # this step's spikes become "previous" next step

        # No self-connections: the forward pass zeroed the W_rec diagonal, so zero the
        # diagonal gradient too -- otherwise the writer wastes pulses on diagonal cells.
        grad_rec.fill_diagonal_(0.0)

        return {
            # loss and gradients share the SAME normalization (/T), so the returned
            # gradient is exactly d(loss)/dW. (For regression n_mask==T anyway.)
            "loss": loss_acc / T,
            "y": ys,
            "grad_rec": grad_rec / T,
            "grad_in": grad_in / T,
            "grad_out": grad_out / T,
            "grad_b": grad_b / T,
            "e_rec_list": e_rec_list,   # raw eligibility traces (None unless return_traces)
            "e_in_list": e_in_list,
        }