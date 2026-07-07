"""
network.py
----------
LSNN (LIF + ALIF karisik) + ONLINE e-prop.

Ileri gecis sirasinda (Bellec 2020):
  - membran/spike:      v, z   (denk. 6,7,9,10)
  - pseudo-derivative:  psi    (Methods)
  - eligibility vector: zbar (=eps_v, F_alpha(z_pre)),  eps_a (ALIF, denk.24)
  - eligibility trace:  e = psi*(zbar - beta*eps_a)     (denk.25)
  - filtreli trace:     ebar = F_kappa(e)               (denk.28, readout leak)
  - learning signal:    L = (y - y*) @ B                (denk.4)
  - gradyan:            grad_W += L * ebar   (t uzerinden toplam)

Gradyan DESCENT icin agirlik guncellemesi: dW = -lr * grad_W.
Bu grad zaten trial boyunca t uzerinden toplandigi icin "episodic" e-prop'tur
(= senin acikca deklare edecegin modelleme tercihi; gradient-accumulation
writer ile birebir ortusur).
"""
from __future__ import annotations
import torch
from neurons import heaviside, pseudo_derivative


class LSNN:
    def __init__(self, ncfg, tcfg, syn_rec, syn_in, W_out, b_out, B_rec,
                 torch_device="cpu", dtype=torch.float32):
        self.nc, self.tc = ncfg, tcfg
        self.syn_rec = syn_rec      # Synapse (W_rec cihazda)
        self.syn_in = syn_in        # Synapse (W_in cihazda)
        self.W_out = W_out          # [n_out, n_rec]
        self.b_out = b_out          # [n_out]
        self.B_rec = B_rec          # [n_out, n_rec] learning-signal feedback
        self.dev = torch_device
        self.dtype = dtype

        n = ncfg.n_rec
        # hangi noronlar ALIF? (ilk adaptive_frac orani)
        n_ad = int(round(ncfg.adaptive_frac * n))
        self.is_adaptive = torch.zeros(n, device=torch_device, dtype=dtype)
        self.is_adaptive[:n_ad] = 1.0
        self.beta_vec = ncfg.beta * self.is_adaptive     # LIF noronlarda beta=0

    def run_trial(self, X, Ystar, accumulate_grads=True):
        """
        X:     [T, n_in]   frozen girdi
        Ystar: [T, n_out]  hedef
        Doner: dict(loss, y, grad_rec, grad_in, grad_out, grad_b)
        """
        nc, tc = self.nc, self.tc
        alpha, rho, kappa = nc.alpha, nc.rho, nc.kappa
        v_th, beta = nc.v_th, self.beta_vec
        n, n_in, n_out = nc.n_rec, tc.n_in, tc.n_out
        T = X.shape[0]
        dev, dt = self.dev, self.dtype

        W_rec = self.syn_rec.weight()            # [n, n]
        W_rec = W_rec - torch.diag(torch.diagonal(W_rec))   # self-connection yok
        W_in = self.syn_in.weight()              # [n, n_in]

        z = torch.zeros(n, device=dev, dtype=dt)
        v = torch.zeros(n, device=dev, dtype=dt)
        a = torch.zeros(n, device=dev, dtype=dt)
        y = torch.zeros(n_out, device=dev, dtype=dt)

        zbar_rec = torch.zeros(n, device=dev, dtype=dt)      # eps_v (recurrent pre)
        zbar_in = torch.zeros(n_in, device=dev, dtype=dt)    # eps_v (input pre)
        epsa_rec = torch.zeros(n, n, device=dev, dtype=dt)   # eps_a [post,pre]
        epsa_in = torch.zeros(n, n_in, device=dev, dtype=dt)
        ebar_rec = torch.zeros(n, n, device=dev, dtype=dt)
        ebar_in = torch.zeros(n, n_in, device=dev, dtype=dt)
        zbar_out = torch.zeros(n, device=dev, dtype=dt)      # readout icin

        grad_rec = torch.zeros(n, n, device=dev, dtype=dt)
        grad_in = torch.zeros(n, n_in, device=dev, dtype=dt)
        grad_out = torch.zeros(n_out, n, device=dev, dtype=dt)
        grad_b = torch.zeros(n_out, device=dev, dtype=dt)

        loss = 0.0
        ys = torch.zeros(T, n_out, device=dev, dtype=dt)

        for t in range(T):
            x_t = X[t]                                   # [n_in]
            # --- membran & spike (onceki z ile) ---
            I = W_rec @ z + W_in @ x_t                   # [n]
            v = alpha * v + I - z * v_th
            A = v_th + beta * a                          # ALIF esigi (LIF: A=v_th)
            z_new = heaviside(v - A)
            psi = pseudo_derivative(v, A, v_th, nc.gamma_pd)   # [n]

            # --- eligibility vektorleri (z_pre = onceki z / x_t) ---
            zbar_rec = alpha * zbar_rec + z              # F_alpha(z^{t-1})
            zbar_in = alpha * zbar_in + x_t
            # eligibility trace (denk.25) - mevcut psi, zbar, eps_a ile
            e_rec = psi[:, None] * (zbar_rec[None, :] - beta[:, None] * epsa_rec)
            e_in = psi[:, None] * (zbar_in[None, :] - beta[:, None] * epsa_in)
            # eps_a guncelle (denk.24)
            epsa_rec = psi[:, None] * zbar_rec[None, :] + (rho - beta[:, None] * psi[:, None]) * epsa_rec
            epsa_in = psi[:, None] * zbar_in[None, :] + (rho - beta[:, None] * psi[:, None]) * epsa_in
            # filtreli eligibility (readout leak, denk.28)
            ebar_rec = kappa * ebar_rec + e_rec
            ebar_in = kappa * ebar_in + e_in

            # --- readout & hata ---
            y = kappa * y + self.W_out @ z_new + self.b_out    # [n_out]
            ys[t] = y
            err = y - Ystar[t]                                 # [n_out]
            loss += 0.5 * float((err ** 2).sum().item())

            if accumulate_grads:
                L = self.B_rec.t() @ err                       # [n] learning signal
                grad_rec += L[:, None] * ebar_rec
                grad_in += L[:, None] * ebar_in
                zbar_out = kappa * zbar_out + z_new
                grad_out += err[:, None] * zbar_out[None, :]
                grad_b += err

            a = rho * a + z_new                                # adaptasyon (denk.10)
            z = z_new

        return {
            "loss": loss / T,
            "y": ys,
            "grad_rec": grad_rec / T,
            "grad_in": grad_in / T,
            "grad_out": grad_out / T,
            "grad_b": grad_b / T,
        }