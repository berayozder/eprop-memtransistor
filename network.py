"""
network.py
----------
LSNN (mixed LIF + ALIF population) + ONLINE e-prop implementation.

During the SNN forward pass (Bellec et al., 2020):
  - Membrane / spike dynamics: v, z   (Equations 6, 7, 9, 10)
  - Pseudo-derivative: psi    (Methods)
  - Eligibility vectors: zbar (= eps_v, F_alpha(z_pre)), eps_a (ALIF feedback trace, Equation 24)
  - Eligibility trace: e = psi * (zbar - beta * eps_a)     (Equation 25)
  - Filtered eligibility trace: ebar = F_kappa(e)          (Equation 28, matching the readout decay rate)
  - Learning signal: L = (y - y*) @ B                      (Equation 4)
  - Gradient accumulation: grad_W += L * ebar              (accumulated over time steps 't')

Gradient descent update step: dW = -lr * grad_W.
Since the gradient is accumulated over 't' during a trial and written at the end of the episode,
this models "episodic" e-prop. This design aligns naturally with the physical gradient-accumulation writer.
"""
from __future__ import annotations
import torch
from neurons import heaviside, pseudo_derivative


class LSNN:
    def __init__(self, ncfg, tcfg, syn_rec, syn_in, W_out, b_out, B_rec,
                 torch_device="cpu", dtype=torch.float32):
        self.nc, self.tc = ncfg, tcfg
        self.syn_rec = syn_rec      # Synapse object (W_rec allocated on device)
        self.syn_in = syn_in        # Synapse object (W_in allocated on device)
        self.W_out = W_out          # Output readout weights [n_out, n_rec]
        self.b_out = b_out          # Output readout biases [n_out]
        self.B_rec = B_rec          # Learning signal feedback matrix [n_out, n_rec]
        self.dev = torch_device
        self.dtype = dtype

        n = ncfg.n_rec
        # Determine adaptive (ALIF) neurons (the first adaptive_frac fraction of the network)
        n_ad = int(round(ncfg.adaptive_frac * n))
        self.is_adaptive = torch.zeros(n, device=torch_device, dtype=dtype)
        self.is_adaptive[:n_ad] = 1.0
        self.beta_vec = ncfg.beta * self.is_adaptive     # LIF neurons have beta=0

    def run_trial(self, X, Ystar, accumulate_grads=True):
        """
        X:     [T, n_in]   Frozen input spike raster
        Ystar: [T, n_out]  Target wave form
        Returns: dict containing trial loss, output y, and gradients (grad_rec, grad_in, grad_out, grad_b)
        """
        nc, tc = self.nc, self.tc
        alpha, rho, kappa = nc.alpha, nc.rho, nc.kappa
        v_th, beta = nc.v_th, self.beta_vec
        n, n_in, n_out = nc.n_rec, tc.n_in, tc.n_out
        T = X.shape[0]
        dev, dt = self.dev, self.dtype

        W_rec = self.syn_rec.weight()            # Recurrent weights [n, n]
        W_rec = W_rec - torch.diag(torch.diagonal(W_rec))   # No self-connections
        W_in = self.syn_in.weight()              # Input weights [n, n_in]

        z = torch.zeros(n, device=dev, dtype=dt)
        v = torch.zeros(n, device=dev, dtype=dt)
        a = torch.zeros(n, device=dev, dtype=dt)
        y = torch.zeros(n_out, device=dev, dtype=dt)

        zbar_rec = torch.zeros(n, device=dev, dtype=dt)      # eps_v (recurrent presynaptic trace)
        zbar_in = torch.zeros(n_in, device=dev, dtype=dt)    # eps_v (input presynaptic trace)
        epsa_rec = torch.zeros(n, n, device=dev, dtype=dt)   # eps_a memory feedback trace [post, pre]
        epsa_in = torch.zeros(n, n_in, device=dev, dtype=dt)
        ebar_rec = torch.zeros(n, n, device=dev, dtype=dt)   # Filtered recurrent trace
        ebar_in = torch.zeros(n, n_in, device=dev, dtype=dt) # Filtered input trace
        zbar_out = torch.zeros(n, device=dev, dtype=dt)      # Post-synaptic trace for readout update

        grad_rec = torch.zeros(n, n, device=dev, dtype=dt)
        grad_in = torch.zeros(n, n_in, device=dev, dtype=dt)
        grad_out = torch.zeros(n_out, n, device=dev, dtype=dt)
        grad_b = torch.zeros(n_out, device=dev, dtype=dt)

        loss = 0.0
        ys = torch.zeros(T, n_out, device=dev, dtype=dt)

        for t in range(T):
            x_t = X[t]                                   # Input spike vector [n_in]
            
            # --- Membrane dynamics & spike generation (using z from step t-1) ---
            I = W_rec @ z + W_in @ x_t                   # Input current [n]
            v = alpha * v + I - z * v_th
            A = v_th + beta * a                          # Adaptive threshold (LIF: A = v_th)
            z_new = heaviside(v - A)
            psi = pseudo_derivative(v, A, v_th, nc.gamma_pd)   # Pseudo-derivative [n]

            # --- Eligibility vectors (z_pre = z or x_t at step t-1) ---
            zbar_rec = alpha * zbar_rec + z              # F_alpha(z^{t-1})
            zbar_in = alpha * zbar_in + x_t
            
            # Recurrent and input eligibility traces (Equation 25)
            e_rec = psi[:, None] * (zbar_rec[None, :] - beta[:, None] * epsa_rec)
            e_in = psi[:, None] * (zbar_in[None, :] - beta[:, None] * epsa_in)
            
            # Update ALIF adaptation traces (Equation 24)
            epsa_rec = psi[:, None] * zbar_rec[None, :] + (rho - beta[:, None] * psi[:, None]) * epsa_rec
            epsa_in = psi[:, None] * zbar_in[None, :] + (rho - beta[:, None] * psi[:, None]) * epsa_in
            
            # Compute filtered eligibility traces (readout leak filter, Equation 28)
            ebar_rec = kappa * ebar_rec + e_rec
            ebar_in = kappa * ebar_in + e_in

            # --- Readout & Loss calculation ---
            y = kappa * y + self.W_out @ z_new + self.b_out    # Output signal [n_out]
            ys[t] = y
            err = y - Ystar[t]                                 # Output error [n_out]
            loss += 0.5 * float((err ** 2).sum().item())

            if accumulate_grads:
                L = self.B_rec.t() @ err                       # Recurrent learning signal [n]
                grad_rec += L[:, None] * ebar_rec
                grad_in += L[:, None] * ebar_in
                zbar_out = kappa * zbar_out + z_new
                grad_out += err[:, None] * zbar_out[None, :]
                grad_b += err

            a = rho * a + z_new                                # ALIF Adaptation variable update (Equation 10)
            z = z_new

        return {
            "loss": loss / T,
            "y": ys,
            "grad_rec": grad_rec / T,
            "grad_in": grad_in / T,
            "grad_out": grad_out / T,
            "grad_b": grad_b / T,
        }